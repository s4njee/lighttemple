// ---------------------------------------------------------------------------
// ltc.wgsl — analytic area lights with linearly transformed cosines (LTC).
//
//   Heitz, Dupuy, Hill & Neubelt 2016, "Real-Time Polygonal-Light Shading with
//   Linearly Transformed Cosines"; Hill & Heitz 2016 (Advances course: LTC Fresnel,
//   the "clipless" sphere approximation); Heitz & Hill 2017, "Real-Time Line- and
//   Disk-Light Shading with Linearly Transformed Cosines" (ellipses).
//
// A GGX lobe (for one view angle and roughness) is the clamped cosine distribution
// D_o warped by a 3×3 matrix M. Integrating the lobe over a polygon P equals
// integrating D_o over M⁻¹P, and a clamped cosine over a polygon has a closed
// form (Lambert's edge sum). So: transform the light's corners by M⁻¹ (from the
// LUT) and run the diffuse formula. The identity matrix gives Lambert diffuse.
//
// Shapes are planar, given by centre `c` and half-axis vectors `ex`, `ey` *relative
// to the shaded point*, already transformed into cosine space (multiply each by
// `LtcLobe.m`, or by `ltc_frame(n, v)` for diffuse). Linear maps keep straight
// edges straight and ellipses elliptical, so transforming c/ex/ey is enough. The
// emitting side is the one cross(ex, ey) points to; `two_sided` lights both.
//
// Every function returns a form factor, ∫ D_o dω over the shape (0..1): multiply
// by the light's radiance (and the lobe's magnitude for specular).
//
// Requires (prepended): common.wgsl (PI, TAU). Textures are passed as arguments
// so callers keep their own bind group layouts; see src/gpu/ltc.ts for the data.
// ---------------------------------------------------------------------------

const LTC_LUT_SIZE: f32 = 64.0;
const LTC_HORIZON_NONE: u32 = 0u;
const LTC_HORIZON_CLIP: u32 = 1u;
const LTC_HORIZON_SPHERE: u32 = 2u;

struct LtcLobe {
  /// World → cosine space: M⁻¹ × the (T1, T2, N) rotation.
  m: mat3x3f,
  /// Specular albedo for F0 = 1 and the Fresnel part: albedo = F0 · magnitude + (1 − F0) · fresnel.
  magnitude: f32,
  fresnel: f32,
};

fn ltc_uv(x: f32, y: f32) -> vec2f {
  return vec2f(x, y) * ((LTC_LUT_SIZE - 1.0) / LTC_LUT_SIZE) + 0.5 / LTC_LUT_SIZE;
}

/// Rotation world → (T1, T2, N), with T1 in the plane of n and v (the LUT's frame).
fn ltc_frame(n: vec3f, v: vec3f) -> mat3x3f {
  var t1 = v - n * dot(v, n);
  let len2 = dot(t1, t1);
  if (len2 > 1e-10) {
    t1 *= inverseSqrt(len2);
  } else {
    // Normal incidence: the lobe is isotropic, any tangent will do.
    let a = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    t1 = normalize(a - n * dot(a, n));
  }
  return transpose(mat3x3f(t1, cross(n, t1), n));
}

/// Fetches the GGX lobe for this view (two bilinear LUT taps).
fn ltc_lobe(lut_matrix: texture_2d<f32>, lut_magnitude: texture_2d<f32>, smp: sampler, n: vec3f, v: vec3f, roughness: f32) -> LtcLobe {
  let uv = ltc_uv(roughness, sqrt(1.0 - saturate(dot(n, v))));
  let t1 = textureSampleLevel(lut_matrix, smp, uv, 0.0);
  let t2 = textureSampleLevel(lut_magnitude, smp, uv, 0.0);
  let minv = mat3x3f(vec3f(t1.x, 0.0, t1.y), vec3f(0.0, 1.0, 0.0), vec3f(t1.z, 0.0, t1.w));
  return LtcLobe(minv * ltc_frame(n, v), t2.x, t2.y);
}

/// Vector form factor of the great arc a → b (unit vectors), divided by 2π: cross(a, b) · θ / sin θ,
/// with Hill & Heitz's rational fit of θ / sin θ (no acos, accurate across the whole range).
fn ltc_edge(a: vec3f, b: vec3f) -> vec3f {
  let x = dot(a, b);
  let y = abs(x);
  let p = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
  let q = 3.4175940 + (4.1616724 + y) * y;
  let v = p / q;
  let theta_sintheta = select(0.5 * inverseSqrt(max(1.0 - x * x, 1e-7)) - v, v, x > 0.0);
  return cross(a, b) * theta_sintheta;
}

/// The "clipless" horizon approximation: replace the shape by the sphere cap with the same
/// vector form factor `f`, whose horizon-clipped form factor is tabulated (magnitude LUT .w).
fn ltc_sphere_clip(lut_magnitude: texture_2d<f32>, smp: sampler, f: vec3f) -> f32 {
  let len = length(f);
  if (len < 1e-8) { return 0.0; }
  let z = f.z / len;
  return len * textureSampleLevel(lut_magnitude, smp, ltc_uv(z * 0.5 + 0.5, saturate(len)), 0.0).w;
}

// --- Polygons: rectangles and regular n-gons ------------------------------------

struct LtcClip {
  first: vec3f,
  prev: vec3f,
  count: u32,
  sum: f32,
};

/// Streaming Sutherland–Hodgman output: integrates each edge as soon as its end vertex is known.
fn ltc_clip_emit(s: ptr<function, LtcClip>, p: vec3f) {
  let q = p * inverseSqrt(max(dot(p, p), 1e-30));
  if ((*s).count == 0u) {
    (*s).first = q;
  } else {
    (*s).sum += ltc_edge((*s).prev, q).z;
  }
  (*s).prev = q;
  (*s).count += 1u;
}

/// Form factor of a polygon with `sides` vertices at c + radius · (cos θ·ex + sin θ·ey),
/// θ = phase − k·2π/sides (clockwise about cross(ex, ey), the winding the edge sum expects).
/// A rectangle is sides = 4, phase = 5π/4, radius = √2. `horizon` picks how the part below
/// the shading plane is removed: exact clipping (up to sides + 1 vertices), the sphere
/// approximation (one extra LUT tap, no branches), or nothing (wrong where a light straddles it).
fn ltc_polygon(
  lut_magnitude: texture_2d<f32>,
  smp: sampler,
  c: vec3f,
  ex: vec3f,
  ey: vec3f,
  sides: u32,
  phase: f32,
  radius: f32,
  horizon: u32,
  two_sided: bool,
) -> f32 {
  // The shaded point (the origin) sees the emitting side when cross(ex, ey) points at it.
  // Any matrix with a positive determinant (M⁻¹ here) preserves the sign.
  let front = dot(cross(ex, ey), c) < 0.0;
  if (!front && !two_sided) { return 0.0; }
  let step = -TAU / f32(sides);
  let rot = vec2f(cos(step), sin(step));
  var cs = vec2f(cos(phase), sin(phase)) * radius;
  var a = c + cs.x * ex + cs.y * ey;
  let first = a;

  if (horizon == LTC_HORIZON_CLIP) {
    var s = LtcClip(vec3f(0.0), vec3f(0.0), 0u, 0.0);
    for (var k = 1u; k <= sides; k++) {
      cs = vec2f(cs.x * rot.x - cs.y * rot.y, cs.x * rot.y + cs.y * rot.x);
      let b = select(c + cs.x * ex + cs.y * ey, first, k == sides);
      if (a.z > 0.0) { ltc_clip_emit(&s, a); }
      if ((a.z > 0.0) != (b.z > 0.0)) { ltc_clip_emit(&s, a + (b - a) * (a.z / (a.z - b.z))); }
      a = b;
    }
    if (s.count > 1u) { s.sum += ltc_edge(s.prev, s.first).z; }
    return select(max(s.sum, 0.0), abs(s.sum), two_sided);
  }

  var f = vec3f(0.0);
  var na = normalize(a);
  let nfirst = na;
  for (var k = 1u; k <= sides; k++) {
    cs = vec2f(cs.x * rot.x - cs.y * rot.y, cs.x * rot.y + cs.y * rot.x);
    let nb = select(normalize(c + cs.x * ex + cs.y * ey), nfirst, k == sides);
    f += ltc_edge(na, nb);
    na = nb;
  }
  // Seen from behind, the winding (and so the vector form factor) flips.
  if (!front) { f = -f; }
  if (horizon == LTC_HORIZON_SPHERE) { return ltc_sphere_clip(lut_magnitude, smp, f); }
  return max(f.z, 0.0);
}

fn ltc_rect(lut_magnitude: texture_2d<f32>, smp: sampler, c: vec3f, ex: vec3f, ey: vec3f, horizon: u32, two_sided: bool) -> f32 {
  return ltc_polygon(lut_magnitude, smp, c, ex, ey, 4u, 1.25 * PI, sqrt(2.0), horizon, two_sided);
}

/// Radius factor that gives a regular n-gon the area of its circumscribing ellipse.
fn ltc_ngon_radius(sides: u32) -> f32 {
  let n = f32(sides);
  return sqrt(TAU / (n * sin(TAU / n)));
}

// --- Ellipses (Heitz & Hill 2017) --------------------------------------------------

/// Real roots of c.w·x³ + c.z·x² + c.y·x + c.x (Blinn 2007, "How to solve a cubic equation",
/// algorithms A and D). The smallest root is returned in .y.
fn ltc_solve_cubic(coef_in: vec4f) -> vec3f {
  var coef = vec4f(coef_in.xyz / coef_in.w, coef_in.w);
  coef = vec4f(coef.x, coef.yz / 3.0, coef.w);
  let ca = coef.w;
  let cb = coef.z;
  let cc = coef.y;
  let cd = coef.x;
  // Hessian and discriminant.
  let delta = vec3f(-coef.z * coef.z + coef.y, -coef.y * coef.z + coef.x, dot(vec2f(coef.z, -coef.y), coef.xy));
  let disc = dot(vec2f(4.0 * delta.x, -delta.y), delta.zy);
  let sq_disc = sqrt(max(disc, 0.0));

  // Algorithm A: the largest root.
  let d_a = -2.0 * cb * delta.x + delta.y;
  let theta_a = atan2(sq_disc, -d_a) / 3.0;
  let x1a = 2.0 * sqrt(max(-delta.x, 0.0)) * cos(theta_a);
  let x3a = 2.0 * sqrt(max(-delta.x, 0.0)) * cos(theta_a + (2.0 / 3.0) * PI);
  let xl = select(x3a, x1a, (x1a + x3a) > 2.0 * cb);
  let xlc = vec2f(xl - cb, ca);

  // Algorithm D: the smallest root.
  let d_d = -cd * delta.y + 2.0 * cc * delta.z;
  let theta_d = atan2(cd * sq_disc, -d_d) / 3.0;
  let x1d = 2.0 * sqrt(max(-delta.z, 0.0)) * cos(theta_d);
  let x3d = 2.0 * sqrt(max(-delta.z, 0.0)) * cos(theta_d + (2.0 / 3.0) * PI);
  let xs = select(x3d, x1d, x1d + x3d < 2.0 * cc);
  let xsc = vec2f(-cd, xs + cc);

  // The middle root from the other two.
  let e = xlc.y * xsc.y;
  let f = -xlc.x * xsc.y - xlc.y * xsc.x;
  let g = xlc.x * xsc.x;
  let xmc = vec2f(cc * f - cb * g, -cb * f + cc * e);
  var root = vec3f(xsc.x / xsc.y, xmc.x / xmc.y, xlc.x / xlc.y);
  if (root.x < root.y && root.x < root.z) {
    root = root.yxz;
  } else if (root.z < root.x && root.z < root.y) {
    root = root.xzy;
  }
  return root;
}

/// Form factor of the ellipse c + cos θ·ex + sin θ·ey (any ex, ey: the transformed disc is a
/// general ellipse). The ellipse subtends an elliptic cone; the cubic's roots give the cone's
/// principal axes, hence the solid angle of an equivalent sphere cap and its average direction,
/// and the horizon is handled by the same clipped-sphere table as the "clipless" polygons.
fn ltc_ellipse(lut_magnitude: texture_2d<f32>, smp: sampler, c: vec3f, ex: vec3f, ey: vec3f, two_sided: bool) -> f32 {
  if (!two_sided && dot(cross(ex, ey), c) >= 0.0) { return 0.0; }
  var v1 = ex;
  var v2 = ey;
  let d11 = dot(v1, v1);
  let d22 = dot(v2, v2);
  let d12 = dot(v1, v2);
  var a: f32;
  var b: f32;
  if (abs(d12) / sqrt(d11 * d22) > 1e-4) {
    // Orthogonalise the axes: eigenvectors of the 2×2 Gram matrix.
    let tr = d11 + d22;
    let det = sqrt(max(d11 * d22 - d12 * d12, 0.0));
    let u = 0.5 * sqrt(max(tr - 2.0 * det, 0.0));
    let w = 0.5 * sqrt(tr + 2.0 * det);
    let e_max = (u + w) * (u + w);
    let e_min = (u - w) * (u - w);
    var v1n: vec3f;
    var v2n: vec3f;
    if (d11 > d22) {
      v1n = d12 * v1 + (e_max - d11) * v2;
      v2n = d12 * v1 + (e_min - d11) * v2;
    } else {
      v1n = d12 * v2 + (e_max - d22) * v1;
      v2n = d12 * v2 + (e_min - d22) * v1;
    }
    a = 1.0 / e_max;
    b = 1.0 / max(e_min, 1e-12);
    v1 = normalize(v1n);
    v2 = normalize(v2n);
  } else {
    a = 1.0 / d11;
    b = 1.0 / d22;
    v1 *= sqrt(a);
    v2 *= sqrt(b);
  }
  var v3 = cross(v1, v2);
  if (dot(c, v3) < 0.0) { v3 = -v3; }
  let l = dot(v3, c);
  // The shaded point lies in the ellipse's plane: it sees an edge-on sliver.
  if (l < 1e-6) { return 0.0; }
  let x0 = dot(v1, c) / l;
  let y0 = dot(v2, c) / l;
  a *= l * l;
  b *= l * l;
  let c0 = a * b;
  let c1 = a * b * (1.0 + x0 * x0 + y0 * y0) - a - b;
  let c2 = 1.0 - a * (1.0 + x0 * x0) - b * (1.0 + y0 * y0);
  let roots = ltc_solve_cubic(vec4f(c0, c1, c2, 1.0));
  let e1 = roots.x;
  let e2 = roots.y;
  let e3 = roots.z;
  let avg = normalize(mat3x3f(v1, v2, v3) * vec3f(a * x0 / (a - e2), b * y0 / (b - e2), 1.0));
  let l1 = sqrt(max(-e2 / e3, 0.0));
  let l2 = sqrt(max(-e2 / e1, 0.0));
  let ff = l1 * l2 * inverseSqrt((1.0 + l1 * l1) * (1.0 + l2 * l2));
  return ff * textureSampleLevel(lut_magnitude, smp, ltc_uv(avg.z * 0.5 + 0.5, saturate(ff)), 0.0).w;
}
