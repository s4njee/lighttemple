// ---------------------------------------------------------------------------
// Area Lights — a product shot lit by rectangular and elliptical soft boxes.
//
//   LTC:        GGX specular + Lambert diffuse integrated analytically over each
//               light (ltc.wgsl): rectangles as polygons, discs as ellipses or n-gons.
//   point:      the same lights collapsed to points of equal power, for comparison.
//   reference:  per-pixel Monte Carlo of the same (unshadowed) integral, light and
//               GGX sampling combined with MIS, accumulated over frames by the host.
//   shadows:    LTC is unshadowed; an optional cone-vs-capsule-proxy estimate dims
//               each light per pixel (UE4 capsule shadows), and the same proxies give
//               contact occlusion for the ambient term.
//
// Outputs premultiplied linear HDR (alpha = coverage) into the post chain.
// Requires (prepended): common.wgsl, fullscreen.wgsl, VertexIn, PROP_WGSL, the
// `Frame` struct, brdf.wgsl and ltc.wgsl.
// ---------------------------------------------------------------------------

struct Light {
  // xyz = centre, w = shape (0 off, 1 rectangle, 2 ellipse)
  center: vec4f,
  // Half-axis vectors (width, height); the light emits along cross(ex, ey). ex.w = area.
  ex: vec4f,
  ey: vec4f,
  // rgb = radiance (linear)
  radiance: vec4f,
};

struct Proxy {
  // xyz = segment start, w = radius
  a: vec4f,
  // xyz = segment end, w = index of the prop it stands in for
  b: vec4f,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var<uniform> lights: array<Light, 3>;
@group(0) @binding(3) var<storage, read> proxies: array<Proxy>;
@group(0) @binding(4) var ltc_matrix: texture_2d<f32>;
@group(0) @binding(5) var ltc_magnitude: texture_2d<f32>;
@group(0) @binding(6) var ltc_sampler: sampler;

const SHAPE_OFF: u32 = 0u;
const SHAPE_ELLIPSE: u32 = 2u;
const MODEL_POINT: u32 = 1u;
const MODEL_REFERENCE: u32 = 2u;
const VIEW_SPECULAR: u32 = 1u;
const VIEW_DIFFUSE: u32 = 2u;
const VIEW_FOOTPRINT: u32 = 3u;
const VIEW_SHADOWS: u32 = 4u;
/// Roughness floor (α ≥ 0.002), shared by every model so they stay comparable.
const MIN_ROUGHNESS: f32 = 0.045;

fn light_shape(l: Light) -> u32 { return u32(l.center.w + 0.5); }
fn light_normal(l: Light) -> vec3f { return normalize(cross(l.ex.xyz, l.ey.xyz)); }

/// Debug colour per light: key amber, strip cyan, top magenta.
fn light_hue(i: u32) -> vec3f {
  switch i {
    case 0u: { return vec3f(1.0, 0.55, 0.12); }
    case 1u: { return vec3f(0.12, 0.7, 1.0); }
    default: { return vec3f(0.9, 0.2, 0.85); }
  }
}

// --- Main pass ---------------------------------------------------------------

struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) id: u32,
};

@vertex
fn vs_main(v: VertexIn, @builtin(instance_index) id: u32) -> VsOut {
  let m = objects[id].model;
  let wp = m * vec4f(v.position, 1.0);
  // Non-uniform scale (the can): transform normals by the inverse scale, i.e. divide by the
  // squared column lengths, which is the inverse-transpose for rotation × scale matrices.
  let s2 = vec3f(dot(m[0].xyz, m[0].xyz), dot(m[1].xyz, m[1].xyz), dot(m[2].xyz, m[2].xyz));
  var out: VsOut;
  out.position = frame.viewProj * wp;
  out.world_pos = wp.xyz;
  out.normal = (m * vec4f(v.normal / s2, 0.0)).xyz;
  out.id = id;
  return out;
}

struct Shade {
  diffuse: vec3f,
  specular: vec3f,
  /// Diffuse form factor (irradiance / (π · radiance)): the light's footprint.
  footprint: f32,
};

/// Analytic LTC integration of one light: two shape integrals (specular in the lobe's cosine
/// space, diffuse in the plain tangent frame), each a handful of edges or one cubic solve.
fn shade_ltc(l: Light, p: vec3f, lobe: LtcLobe, rot: mat3x3f, spec_albedo: vec3f, diffuse_albedo: vec3f) -> Shade {
  let c = l.center.xyz - p;
  let ex = l.ex.xyz;
  let ey = l.ey.xyz;
  let two_sided = frame.twoSided != 0u;
  var spec_ff: f32;
  var diff_ff: f32;
  if (light_shape(l) == SHAPE_ELLIPSE) {
    if (frame.discSides == 0u) {
      spec_ff = ltc_ellipse(ltc_magnitude, ltc_sampler, lobe.m * c, lobe.m * ex, lobe.m * ey, two_sided);
      diff_ff = ltc_ellipse(ltc_magnitude, ltc_sampler, rot * c, rot * ex, rot * ey, two_sided);
    } else {
      let r = ltc_ngon_radius(frame.discSides);
      spec_ff = ltc_polygon(ltc_magnitude, ltc_sampler, lobe.m * c, lobe.m * ex, lobe.m * ey, frame.discSides, 0.0, r, frame.horizon, two_sided);
      diff_ff = ltc_polygon(ltc_magnitude, ltc_sampler, rot * c, rot * ex, rot * ey, frame.discSides, 0.0, r, frame.horizon, two_sided);
    }
  } else {
    spec_ff = ltc_rect(ltc_magnitude, ltc_sampler, lobe.m * c, lobe.m * ex, lobe.m * ey, frame.horizon, two_sided);
    diff_ff = ltc_rect(ltc_magnitude, ltc_sampler, rot * c, rot * ex, rot * ey, frame.horizon, two_sided);
  }
  return Shade(l.radiance.rgb * diffuse_albedo * diff_ff, l.radiance.rgb * spec_albedo * spec_ff, diff_ff);
}

/// brdf.wgsl's `brdf_direct`, with the two lobes kept apart (both already × N·L).
struct BrdfSplit {
  diffuse: vec3f,
  specular: vec3f,
};

fn brdf_split(m: Material, n: vec3f, v: vec3f, l: vec3f) -> BrdfSplit {
  let nol = saturate(dot(n, l));
  if (nol <= 0.0) { return BrdfSplit(vec3f(0.0), vec3f(0.0)); }
  let h = normalize(v + l);
  let nov = max(dot(n, v), 1e-4);
  let a = m.roughness * m.roughness;
  let f = f_schlick(material_f0(m), saturate(dot(v, h)));
  let spec = d_ggx(saturate(dot(n, h)), a) * v_smith_ggx_correlated(nov, nol, a) * f;
  let kd = (vec3f(1.0) - f) * (1.0 - m.metallic);
  return BrdfSplit(kd * m.albedo / PI * nol, spec * nol);
}

/// The same light as a point of equal power at its centre: radiant intensity L·A·cos θ_light.
fn shade_point(l: Light, p: vec3f, n: vec3f, v: vec3f, m: Material) -> Shade {
  let to = l.center.xyz - p;
  let d2 = max(dot(to, to), 1e-6);
  let wi = to * inverseSqrt(d2);
  var cos_l = -dot(light_normal(l), wi);
  if (frame.twoSided != 0u) { cos_l = abs(cos_l); }
  let e = l.ex.w * max(cos_l, 0.0) / d2;
  let b = brdf_split(m, n, v, wi);
  return Shade(b.diffuse * l.radiance.rgb * e, b.specular * l.radiance.rgb * e, e * saturate(dot(n, wi)) / PI);
}

// --- Monte Carlo reference ------------------------------------------------------

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rand(seed: ptr<function, u32>) -> f32 {
  *seed = pcg(*seed);
  return f32(*seed >> 8u) / 16777216.0;
}

/// Density of the BRDF sampling strategy below (GGX half vectors with probability p_spec, else cosine).
fn brdf_pdf(m: Material, n: vec3f, v: vec3f, l: vec3f, p_spec: f32) -> f32 {
  let nol = dot(n, l);
  if (nol <= 0.0) { return 0.0; }
  let h = normalize(v + l);
  let noh = saturate(dot(n, h));
  let a = m.roughness * m.roughness;
  return p_spec * d_ggx(noh, a) * noh / (4.0 * max(dot(v, h), 1e-4)) + (1.0 - p_spec) * nol / PI;
}

/// Distance along o + t·d to the light's surface, or -1 on a miss.
fn light_hit(l: Light, o: vec3f, d: vec3f) -> f32 {
  let nl = light_normal(l);
  let denom = dot(d, nl);
  if (abs(denom) < 1e-6) { return -1.0; }
  let t = dot(l.center.xyz - o, nl) / denom;
  let q = o + d * t - l.center.xyz;
  let uv = vec2f(dot(q, l.ex.xyz) / dot(l.ex.xyz, l.ex.xyz), dot(q, l.ey.xyz) / dot(l.ey.xyz, l.ey.xyz));
  let inside = select(max(abs(uv.x), abs(uv.y)) <= 1.0, dot(uv, uv) <= 1.0, light_shape(l) == SHAPE_ELLIPSE);
  return select(-1.0, t, inside && t > 0.0);
}

/// One light sample + one BRDF sample per iteration, combined with the balance heuristic:
/// each contributes f · L / (pdf_light + pdf_brdf), both pdfs in solid angle.
fn shade_reference(l: Light, p: vec3f, n: vec3f, v: vec3f, m: Material, rot: mat3x3f, seed: ptr<function, u32>) -> Shade {
  let nl = light_normal(l);
  let area = l.ex.w;
  let two_sided = frame.twoSided != 0u;
  let ellipse = light_shape(l) == SHAPE_ELLIPSE;
  let p_spec = mix(0.5, 1.0, m.metallic);
  let a = m.roughness * m.roughness;
  let v_local = rot * v;
  let to_world = transpose(rot);
  var diffuse = vec3f(0.0);
  var specular = vec3f(0.0);
  var footprint = 0.0;
  let count = max(frame.refSamples, 1u);
  for (var i = 0u; i < count; i++) {
    // 1. A uniformly distributed point on the light.
    var u = vec2f(rand(seed), rand(seed));
    var q: vec3f;
    if (ellipse) {
      let r = sqrt(u.x);
      q = l.center.xyz + r * cos(TAU * u.y) * l.ex.xyz + r * sin(TAU * u.y) * l.ey.xyz;
    } else {
      q = l.center.xyz + (2.0 * u.x - 1.0) * l.ex.xyz + (2.0 * u.y - 1.0) * l.ey.xyz;
    }
    let w = q - p;
    let d2 = dot(w, w);
    let wi = w * inverseSqrt(d2);
    var cos_l = -dot(nl, wi);
    if (two_sided) { cos_l = abs(cos_l); }
    if (cos_l > 0.0 && dot(n, wi) > 0.0) {
      let pdf_l = d2 / (area * cos_l);
      let inv = 1.0 / (pdf_l + brdf_pdf(m, n, v, wi, p_spec));
      let b = brdf_split(m, n, v, wi);
      diffuse += b.diffuse * inv;
      specular += b.specular * inv;
      footprint += dot(n, wi) / PI * inv;
    }

    // 2. A direction from the BRDF, kept if it hits the light.
    u = vec2f(rand(seed), rand(seed));
    var wl: vec3f;
    if (rand(seed) < p_spec) {
      let ct = sqrt((1.0 - u.x) / (1.0 + (a * a - 1.0) * u.x));
      let st = sqrt(max(1.0 - ct * ct, 0.0));
      let h = vec3f(st * cos(TAU * u.y), st * sin(TAU * u.y), ct);
      wl = reflect(-v_local, h);
    } else {
      let r = sqrt(u.x);
      wl = vec3f(r * cos(TAU * u.y), r * sin(TAU * u.y), sqrt(max(1.0 - u.x, 0.0)));
    }
    let wi2 = to_world * wl;
    if (wl.z > 0.0) {
      let t = light_hit(l, p, wi2);
      var cos_l2 = -dot(nl, wi2);
      if (two_sided) { cos_l2 = abs(cos_l2); }
      if (t > 0.0 && cos_l2 > 0.0) {
        let pdf_l = t * t / (area * cos_l2);
        let inv = 1.0 / (pdf_l + brdf_pdf(m, n, v, wi2, p_spec));
        let b = brdf_split(m, n, v, wi2);
        diffuse += b.diffuse * inv;
        specular += b.specular * inv;
        footprint += wl.z / PI * inv;
      }
    }
  }
  let k = 1.0 / f32(count);
  return Shade(diffuse * l.radiance.rgb * k, specular * l.radiance.rgb * k, footprint * k);
}

// --- Approximate shadows and occlusion from capsule proxies ---------------------------

/// Point on segment a–b closest to the line through o along unit d.
fn closest_on_segment(a: vec3f, b: vec3f, o: vec3f, d: vec3f) -> vec3f {
  let ab = b - a;
  let w = a - o;
  let dd = dot(ab, d);
  let e = dot(ab, ab);
  let s = select(0.0, saturate((dd * dot(w, d) - dot(w, ab)) / max(e - dd * dd, 1e-6)), e > 1e-8);
  return a + ab * s;
}

/// Fraction of the smaller of two cones (half-angles a1, a2, axes `between` apart) covered by
/// the other: 1 when nested, 0 when disjoint, smooth in between (Oat & Sander 2007 / UE4).
fn cone_overlap(a1: f32, a2: f32, between: f32) -> f32 {
  let diff = abs(a1 - a2);
  let t = saturate((between - diff) / max(a1 + a2 - diff, 1e-4));
  return 1.0 - smoothstep(0.0, 1.0, t);
}

/// How much of the light's cone (its solid angle from p, as a cap) the proxies block.
fn proxy_visibility(p: vec3f, l: Light, self_id: u32) -> f32 {
  let to = l.center.xyz - p;
  let d = length(to);
  let dir = to / d;
  // Ω ≈ A·cos θ / d², as a cap: Ω = 2π(1 − cos α). (Solid angles below are in units of 2π.)
  let cos_al = clamp(1.0 - l.ex.w * max(abs(dot(light_normal(l), dir)), 0.1) / (TAU * d * d), -1.0, 1.0);
  let angle_l = acos(cos_al);
  let omega_l = max(1.0 - cos_al, 1e-6);
  var vis = 1.0;
  for (var i = 0u; i < frame.proxyCount; i++) {
    let pr = proxies[i];
    if (u32(pr.b.w) == self_id) { continue; }
    let s = closest_on_segment(pr.a.xyz, pr.b.xyz, p, dir);
    let to_o = s - p;
    let along = dot(to_o, dir);
    let dist = length(to_o);
    let r = pr.a.w;
    if (along <= 0.0 || along > d || dist <= r) { continue; }
    let angle_o = asin(r / dist);
    let between = acos(clamp(along / dist, -1.0, 1.0));
    let covered = cone_overlap(angle_l, angle_o, between) * (1.0 - cos(min(angle_l, angle_o))) / omega_l;
    vis *= 1.0 - saturate(covered);
  }
  return vis;
}

/// Ambient occlusion: each proxy's cosine-weighted solid angle, (n · d̂)(r / |d|)², from its nearest point.
fn proxy_ao(p: vec3f, n: vec3f, self_id: u32) -> f32 {
  var occ = 0.0;
  for (var i = 0u; i < frame.proxyCount; i++) {
    let pr = proxies[i];
    if (u32(pr.b.w) == self_id) { continue; }
    let ab = pr.b.xyz - pr.a.xyz;
    let s = pr.a.xyz + ab * saturate(dot(p - pr.a.xyz, ab) / max(dot(ab, ab), 1e-8));
    let d = s - p;
    let l2 = max(dot(d, d), 1e-6);
    occ += saturate(dot(n, d) * inverseSqrt(l2)) * pr.a.w * pr.a.w / l2;
  }
  return saturate(1.0 - occ);
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4f {
  let obj = objects[v.id];
  let is_sweep = obj.material.y > 0.5;
  let p = v.world_pos;
  let vdir = normalize(frame.cameraPos - p);
  var n = normalize(v.normal);
  // Interpolated normals can turn away from the eye at silhouettes; the LUT stops at 90°.
  let nv = dot(n, vdir);
  if (nv < 0.02) { n = normalize(n + vdir * (0.02 - nv)); }

  var roughness = obj.color.a;
  if (frame.roughOverride >= 0.0 && !is_sweep) { roughness = frame.roughOverride; }
  let m = Material(obj.color.rgb, max(roughness, MIN_ROUGHNESS), obj.material.x);
  let f0 = material_f0(m);
  let rot = ltc_frame(n, vdir);
  let lobe = ltc_lobe(ltc_matrix, ltc_magnitude, ltc_sampler, n, vdir, m.roughness);
  // LTC's diffuse is plain Lambert: weight it by what the specular lobe leaves over.
  let spec_albedo = f0 * lobe.magnitude + (vec3f(1.0) - f0) * lobe.fresnel;
  let diffuse_albedo = m.albedo * (1.0 - m.metallic) * (vec3f(1.0) - spec_albedo);

  var seed = pcg(u32(v.position.x) + pcg(u32(v.position.y) + pcg(frame.frameIndex)));
  var diffuse = vec3f(0.0);
  var specular = vec3f(0.0);
  var footprint = vec3f(0.0);
  var footprint_total = 0.0;
  var shadow_view = vec3f(1.0);
  for (var i = 0u; i < 3u; i++) {
    let l = lights[i];
    if (light_shape(l) == SHAPE_OFF) { continue; }
    var s: Shade;
    switch frame.lightModel {
      case MODEL_POINT: { s = shade_point(l, p, n, vdir, m); }
      case MODEL_REFERENCE: { s = shade_reference(l, p, n, vdir, m, rot, &seed); }
      default: { s = shade_ltc(l, p, lobe, rot, spec_albedo, diffuse_albedo); }
    }
    var vis = 1.0;
    if (frame.shadows != 0u) { vis = proxy_visibility(p, l, v.id); }
    diffuse += s.diffuse * vis;
    specular += s.specular * vis;
    footprint += light_hue(i) * s.footprint;
    footprint_total += s.footprint;
    // Each light's shadow tinted with its hue (overlaps multiply).
    shadow_view *= mix(light_hue(i) * 0.55, vec3f(1.0), vis);
  }

  // Studio ambient: a dim hemisphere (lit sweep below, dark rig above), occluded by the proxies.
  var ao = 1.0;
  if (frame.shadows != 0u) { ao = proxy_ao(p, n, v.id); }
  let r = reflect(-vdir, n);
  let edge = mix(0.05, 0.8, m.roughness);
  let irradiance = mix(frame.groundColor.rgb, frame.skyColor.rgb, n.y * 0.5 + 0.5) * ao;
  let env = mix(frame.groundColor.rgb, frame.skyColor.rgb, smoothstep(-edge, edge, r.y)) * ao;
  let ab = env_brdf_approx(max(dot(n, vdir), 1e-4), m.roughness);
  let ambient_spec = env * (f0 * ab.x + ab.y);
  let ambient_diffuse = irradiance * m.albedo * (1.0 - m.metallic);

  // Iso-lines of the total footprint, every factor of 2.
  let level = log2(max(footprint_total, 1e-5));
  let iso = 1.0 - saturate(abs(fract(level + 0.5) - 0.5) / max(fwidth(level), 1e-4));

  var col = diffuse + specular + ambient_diffuse + ambient_spec;
  switch frame.view {
    case VIEW_SPECULAR: { col = specular + ambient_spec; }
    case VIEW_DIFFUSE: { col = diffuse + ambient_diffuse; }
    case VIEW_FOOTPRINT: { col = sqrt(footprint) * 1.6 * (1.0 - 0.6 * iso) + vec3f(0.02); }
    case VIEW_SHADOWS: { col = shadow_view * (0.25 + 0.75 * ao) * 0.85; }
    default: {}
  }

  // Coverage: the sweep dissolves into the background colour at its sides and front.
  var coverage = 1.0;
  if (is_sweep && frame.view <= VIEW_DIFFUSE) {
    coverage = (1.0 - smoothstep(frame.fadeX.x, frame.fadeX.y, abs(p.x))) * (1.0 - smoothstep(frame.fadeZ.x, frame.fadeZ.y, p.z));
  }
  return vec4f(col * coverage, coverage);
}

// --- Emitters: each soft box's diffuser and black housing ----------------------------

struct EmitterOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  // z of the local normal: +1 on the diffuser, -1 on the back, 0 on the sides.
  @location(2) face: f32,
  @location(3) @interpolate(flat) id: u32,
};

/// Housing mesh (prism, local x/y across the light, z from -1 at the back to 0 at the
/// diffuser) placed on the light; ey.w is the housing depth.
@vertex
fn vs_emitter(v: VertexIn, @builtin(instance_index) id: u32) -> EmitterOut {
  let l = lights[id];
  let forward = light_normal(l);
  let s = frame.emitterScale;
  let wp = l.center.xyz + (v.position.x * l.ex.xyz + v.position.y * l.ey.xyz) * s + forward * (v.position.z * l.ey.w * s);
  var out: EmitterOut;
  // Switched-off lights collapse outside the clip volume.
  out.position = select(frame.viewProj * vec4f(wp, 1.0), vec4f(2.0, 2.0, 2.0, 1.0), light_shape(l) == SHAPE_OFF);
  out.world_pos = wp;
  out.normal = v.normal.x * normalize(l.ex.xyz) + v.normal.y * normalize(l.ey.xyz) + v.normal.z * forward;
  out.face = v.normal.z;
  out.id = id;
  return out;
}

@fragment
fn fs_emitter(e: EmitterOut) -> @location(0) vec4f {
  let l = lights[e.id];
  // The diffuser emits the light's radiance (what the LTC integral assumes, uniform); a
  // two-sided light has no housing, and its back face emits too.
  if (e.face > 0.5 || (e.face < -0.5 && frame.twoSided != 0u)) { return vec4f(l.radiance.rgb, 1.0); }
  // Black fabric, faintly lit from above so the box reads as a shape.
  let n = normalize(e.normal);
  return vec4f(vec3f(0.006) + vec3f(0.02) * saturate(n.y * 0.5 + 0.5), 1.0);
}

// --- Debug overlay: the two LTC tables, one channel per tile ---------------------------

/// Display-space ramp: positive values dark violet → magenta → amber, negative ones blue.
fn lut_ramp(x: f32) -> vec3f {
  let t = saturate(abs(x));
  if (x < 0.0) { return mix(vec3f(0.04, 0.05, 0.1), vec3f(0.35, 0.7, 1.0), t); }
  let lo = mix(vec3f(0.05, 0.02, 0.12), vec3f(0.72, 0.16, 0.45), smoothstep(0.0, 0.55, t));
  return mix(lo, vec3f(1.0, 0.86, 0.4), smoothstep(0.5, 1.0, t));
}

/// Drawn straight onto the display target after the post chain, so tonemapping can't skew it.
@fragment
fn fs_lut(v: FullscreenOut) -> @location(0) vec4f {
  let grid = v.uv * 2.0;
  let tile = min(vec2u(grid), vec2u(1u));
  let uv = grid - vec2f(tile);
  // 2×2 tiles with a thin gutter, read texel-exact; y up = larger index, like the papers' plots.
  if (any(uv < vec2f(0.015)) || any(uv > vec2f(0.985))) { return vec4f(0.02, 0.02, 0.02, 1.0); }
  let px = clamp(vec2i(vec2f(uv.x, 1.0 - uv.y) * LTC_LUT_SIZE), vec2i(0), vec2i(i32(LTC_LUT_SIZE) - 1));
  let ch = tile.y * 2u + tile.x;
  var x: f32;
  if (frame.lutView == 1u) {
    // Inverse matrix a, b / c, d, each divided by its largest magnitude in the table.
    x = textureLoad(ltc_matrix, px, 0)[ch] / vec4f(1.0, 0.3, 0.6, 1.66)[ch];
  } else {
    // Magnitude n, Fresnel f / total albedo n + f, clipped-sphere ratio (on its own axes).
    let t = textureLoad(ltc_magnitude, px, 0);
    x = select(t[ch], t.x + t.y, ch == 2u);
  }
  return vec4f(lut_ramp(x), 1.0);
}

// --- Progressive accumulation for the reference view ------------------------------------

@group(0) @binding(8) var acc_current: texture_2d<f32>;
@group(0) @binding(9) var acc_history: texture_2d<f32>;

@fragment
fn fs_accumulate(v: FullscreenOut) -> @location(0) vec4f {
  let px = vec2i(v.position.xy);
  return mix(textureLoad(acc_history, px, 0), textureLoad(acc_current, px, 0), frame.accumWeight);
}
