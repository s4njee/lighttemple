// ---------------------------------------------------------------------------
// Lantern Room: omnidirectional point-light shadows from a cube depth map (or
// a dual-paraboloid map, for comparison), filtered in 3D.
// Outputs premultiplied linear HDR (alpha = coverage) into the post chain.
// Requires (prepended): common.wgsl, fullscreen.wgsl, noise.wgsl, brdf.wgsl,
// VertexIn, the `Object` struct, and the `Frame` and `Face` structs.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var shadow_cube: texture_depth_cube;
@group(0) @binding(3) var shadow_dp: texture_depth_2d_array;
@group(0) @binding(4) var shadow_cmp: sampler_comparison;
@group(0) @binding(5) var shadow_point: sampler;

// Material ids (Object.material.y).
const MAT_PLAIN = 0u;
const MAT_ROOM = 1u;
const MAT_FLAME = 2u;
const MAT_RUG = 3u;
const MAT_PAINTING = 4u;

const TECH_CUBE = 0u;

const POISSON_16 = array<vec2f, 16>(
  vec2f(-0.94201624, -0.39906216), vec2f(0.94558609, -0.76890725),
  vec2f(-0.09418410, -0.92938870), vec2f(0.34495938, 0.29387760),
  vec2f(-0.91588581, 0.45771432), vec2f(-0.81544232, -0.87912464),
  vec2f(-0.38277543, 0.27676845), vec2f(0.97484398, 0.75648379),
  vec2f(0.44323325, -0.97511554), vec2f(0.53742981, -0.47373420),
  vec2f(-0.26496911, -0.41893023), vec2f(0.79197514, 0.19090188),
  vec2f(-0.24188840, 0.99706507), vec2f(-0.81409955, 0.91437590),
  vec2f(0.19984126, 0.78641367), vec2f(0.14383161, -0.14100790),
);

// --- Shadow passes -----------------------------------------------------------

@group(0) @binding(6) var<uniform> sface: Face;

/// Cube face: an ordinary 90° perspective render (6 of these per frame).
@vertex
fn vs_cube_shadow(v: VertexIn, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return sface.viewProj * objects[id].model * vec4f(v.position, 1.0);
}

struct DpOut {
  @builtin(position) position: vec4f,
  // > 0 inside this hemisphere.
  @location(0) side: f32,
};

/// Dual paraboloid: the hemisphere around ±y is flattened onto a disc by the
/// paraboloid mapping uv = d.xz / (1 + |d.y|). It is non-linear, so straight
/// edges should bend; the rasteriser draws them straight between vertices,
/// which is why big triangles need tessellating.
@vertex
fn vs_dp_shadow(v: VertexIn, @builtin(instance_index) id: u32) -> DpOut {
  let wp = (objects[id].model * vec4f(v.position, 1.0)).xyz;
  let q = dp_local(wp - sface.lightPos, sface.hemisphere);
  let dist = length(q);
  let n = q / max(dist, 1e-6);
  var out: DpOut;
  // The mapping is continuous across the seam, so triangles straddling it are
  // fine; only vertices deep in the other hemisphere blow up (1 + z -> 0), so
  // clamp those and let the fragment stage discard what lies beyond the seam.
  out.position = vec4f(n.xy / (1.0 + max(n.z, -0.9)), (dist - sface.near) / (sface.far - sface.near), 1.0);
  out.side = n.z;
  return out;
}

@fragment
fn fs_dp_shadow(v: DpOut) {
  if (v.side < 0.0) { discard; }
}

/// World offset from the light -> paraboloid-local frame (z = hemisphere axis).
fn dp_local(d: vec3f, hemisphere: f32) -> vec3f {
  return vec3f(d.x, d.z * hemisphere, d.y * hemisphere);
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
  var out: VsOut;
  out.position = frame.viewProj * wp;
  out.world_pos = wp.xyz;
  // Scales are uniform or axis-aligned on boxes, so normalising m·n is enough.
  out.normal = normalize((m * vec4f(v.normal, 0.0)).xyz);
  out.id = id;
  return out;
}

/// Depth the cube map stores for a point whose largest light-space axis is `ma`
/// (the 90° perspective of whichever face that axis selects).
fn cube_depth(ma: f32) -> f32 {
  let n = frame.shadowNear;
  let f = frame.shadowFar;
  return f / (f - n) * (1.0 - n / ma);
}

fn max_axis(v: vec3f) -> f32 { return max(abs(v.x), max(abs(v.y), abs(v.z))); }

/// One hardware-filtered (2×2 bilinear PCF) comparison: is the point at
/// distance `dist` from the light, along direction `dir`, lit?
/// Each tap picks its own cube face, and its reference depth is computed for
/// that face, so taps that wander across a cube edge stay seamless.
fn shadow_tap(dir: vec3f, dist: f32) -> f32 {
  if (frame.technique == TECH_CUBE) {
    let ma = max_axis(dir) * dist;
    return textureSampleCompareLevel(shadow_cube, shadow_cmp, dir, cube_depth(ma));
  }
  let hemi = select(-1.0, 1.0, dir.y >= 0.0);
  let q = dp_local(dir, hemi);
  let uv = q.xy / (1.0 + q.z);
  let ref_depth = (dist - frame.shadowNear) / (frame.shadowFar - frame.shadowNear);
  return textureSampleCompareLevel(shadow_dp, shadow_cmp, vec2f(0.5 + 0.5 * uv.x, 0.5 - 0.5 * uv.y), select(1, 0, hemi > 0.0), ref_depth);
}

/// Distance from the light to whatever the map recorded along `dir`.
fn blocker_distance(dir: vec3f) -> f32 {
  let n = frame.shadowNear;
  let f = frame.shadowFar;
  if (frame.technique == TECH_CUBE) {
    let z = textureSampleLevel(shadow_cube, shadow_point, dir, 0);
    let ma = n / max(1.0 - z * (f - n) / f, 1e-5);
    return ma / max_axis(dir);
  }
  let hemi = select(-1.0, 1.0, dir.y >= 0.0);
  let q = dp_local(dir, hemi);
  let uv = q.xy / (1.0 + q.z);
  let z = textureSampleLevel(shadow_dp, shadow_point, vec2f(0.5 + 0.5 * uv.x, 0.5 - 0.5 * uv.y), select(1, 0, hemi > 0.0), 0);
  return n + z * (f - n);
}

/// Orthonormal basis around unit vector n (Duff et al. 2017).
fn basis(n: vec3f) -> mat2x3f {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return mat2x3f(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y));
}

struct ShadowQuery {
  // Receiver point (already normal-offset) and its geometric normal.
  p: vec3f,
  n: vec3f,
  // Unit direction light -> receiver, and the distance.
  dir: vec3f,
  dist: f32,
  // World size of one shadow texel at the receiver.
  texel: f32,
  tangents: mat2x3f,
};

/// A tap displaced by `offset` (world units, on the disc perpendicular to the
/// light ray at the receiver). The ray through the displaced point is followed
/// back to the receiver's tangent plane, and that hit distance becomes the
/// reference depth: a wide kernel then never shadows its own flat surface.
fn disc_tap(s: ShadowQuery, offset: vec2f, bias: f32) -> f32 {
  let aim = s.dir * s.dist + s.tangents * offset;
  let dir = normalize(aim);
  let denom = dot(dir, s.n);
  // dot(light -> receiver, n) is negative on lit surfaces.
  let plane = dot(s.dir * s.dist, s.n);
  let dist = select(s.dist, clamp(plane / denom, 0.0, 2.0 * s.dist), denom < -1e-3);
  return shadow_tap(dir, dist - bias);
}

fn point_shadow(world_pos: vec3f, n: vec3f, ndl: f32, frag: vec2f) -> f32 {
  if (frame.filterMode == 0u) { return 1.0; }
  let to_p = world_pos - frame.lightPos;
  let d0 = length(to_p);
  // A 90° face spans 2·(major axis) world units across `mapSize` texels.
  let texel0 = 2.0 * max_axis(to_p) / frame.mapSize;
  // Normal-offset bias, larger at grazing angles where acne starts.
  let p = world_pos + n * frame.normalOffset * texel0 * (1.5 - ndl);
  let v = p - frame.lightPos;
  let dist = length(v);
  var s: ShadowQuery;
  s.p = p;
  s.n = n;
  s.dir = v / dist;
  s.dist = dist;
  s.texel = texel0;
  s.tangents = basis(s.dir);
  let bias = frame.depthBias * texel0;
  // Rotating the noise per frame lets TAA integrate it into a smooth result.
  let angle = ign(frag + frame.noiseOffset) * TAU;
  let rot = mat2x2f(cos(angle), sin(angle), -sin(angle), cos(angle));
  var disk = POISSON_16;

  if (frame.filterMode == 1u) { return shadow_tap(s.dir, dist - bias); }

  var radius = frame.pcfRadius * s.texel;
  if (frame.filterMode == 3u) {
    // PCSS for a spherical light of radius R. A blocker at distance b widens
    // into a penumbra of R·(d − b)/b at the receiver, so search the disc that
    // blockers down to ~12% of the way could reach.
    let r = frame.lightRadius;
    let b_min = max(0.12 * dist, frame.shadowNear * 2.0);
    let search = clamp(r * (dist - b_min) / b_min, 2.0 * s.texel, 0.4 * dist);
    var sum = 0.0;
    var count = 0.0;
    for (var i = 0; i < 16; i++) {
      let off = rot * disk[i] * search;
      let dir = normalize(s.dir * dist + s.tangents * off);
      let b = blocker_distance(dir);
      if (b < dist - bias - s.texel) {
        sum += b;
        count += 1.0;
      }
    }
    if (count == 0.0) { return 1.0; }
    let b = sum / count;
    radius = clamp(r * (dist - b) / b, s.texel, search);
  }

  var vis = 0.0;
  for (var i = 0; i < 16; i++) {
    vis += disc_tap(s, rot * disk[i] * radius, bias);
  }
  return vis / 16.0;
}

// --- Materials ---------------------------------------------------------------

struct Surface {
  albedo: vec3f,
  roughness: f32,
  metallic: f32,
  emissive: vec3f,
};

fn room_surface(p: vec3f, n: vec3f) -> Surface {
  var s = Surface(vec3f(0.0), 0.85, 0.0, vec3f(0.0));
  if (n.y > 0.5) {
    // Floor: planks along x, staggered ends, per-plank tone and grain.
    let w = 0.19;
    let row = floor(p.z / w);
    let len = 1.6;
    let col = floor(p.x / len + hash21(vec2f(row, 3.1)));
    let id = vec2f(row, col);
    let tone = hash21(id);
    let grain = fbm(vec2f(p.x * 1.2, p.z * 22.0) + id * 7.0, 4);
    var c = mix(vec3f(0.30, 0.17, 0.09), vec3f(0.45, 0.27, 0.15), tone) * (0.85 + 0.25 * grain);
    let fz = fract(p.z / w);
    let fx = fract(p.x / len + hash21(vec2f(row, 3.1)));
    let seam = min(min(fz, 1.0 - fz) * w, min(fx, 1.0 - fx) * len);
    c *= mix(0.45, 1.0, smoothstep(0.0, 0.006, seam));
    s.albedo = srgb_to_linear(c * 1.6) * frame.floorTint.rgb;
    s.roughness = 0.55 + 0.2 * tone;
    return s;
  }
  if (n.y < -0.5) {
    // Ceiling: plaster between dark beams.
    let beam = abs(fract(p.x / 1.4) - 0.5) * 1.4;
    s.albedo = select(vec3f(0.55, 0.52, 0.48), vec3f(0.12, 0.07, 0.04), beam < 0.09);
    return s;
  }
  // Walls: limewash above a panelled wainscot and a chair rail.
  let along = select(p.x, p.z, abs(n.x) > 0.5);
  let plaster = 0.92 + 0.08 * fbm(vec2f(along, p.y) * 3.0, 4);
  var c = frame.wallTint.rgb * plaster;
  if (p.y < 0.95) {
    let panel = abs(fract(along / 0.6) - 0.5) * 0.6;
    let groove = smoothstep(0.012, 0.02, panel) * smoothstep(0.02, 0.03, abs(p.y - 0.47));
    c = vec3f(0.16, 0.2, 0.17) * mix(0.6, 1.0, groove);
    s.roughness = 0.5;
  } else if (p.y < 1.0) {
    c = vec3f(0.24, 0.16, 0.1);
    s.roughness = 0.45;
  }
  s.albedo = c;

  // A night window on the back wall: emissive panes behind mullions.
  if (n.z > 0.5) {
    let q = (vec2f(p.x, p.y) - frame.window.xy) / frame.window.zw;
    let e = max(abs(q.x), abs(q.y));
    if (e < 1.08) {
      s.albedo = vec3f(0.08, 0.05, 0.03);
      s.roughness = 0.5;
    }
    let mull = min(abs(q.x), abs(q.y) * frame.window.w / frame.window.z);
    if (e < 1.0 && mull > 0.03) {
      // Moonlit sky: deeper at the top, a little haze near the sill.
      let sky = mix(vec3f(0.035, 0.06, 0.12), vec3f(0.006, 0.01, 0.03), q.y * 0.5 + 0.5);
      let star = step(0.997, hash21(floor((q + 2.0) * 60.0))) * 0.4;
      s.albedo = vec3f(0.02);
      s.roughness = 0.05;
      s.emissive = (sky + star) * frame.windowGlow;
    }
  }
  return s;
}

fn rug_surface(p: vec3f, base: vec3f) -> vec3f {
  let q = abs(p.xz - frame.rug.xy) / frame.rug.zw;
  let edge = max(q.x, q.y);
  var c = base;
  if (edge > 0.82) { c = vec3f(0.55, 0.42, 0.25); }
  if (edge > 0.86 && edge < 0.9) { c = base * 0.5; }
  // Diamond lattice in the field.
  let d = abs(fract(vec2f(q.x * 4.0 + q.y * 4.0, q.x * 4.0 - q.y * 4.0)) - 0.5);
  if (edge < 0.78 && min(d.x, d.y) < 0.06) { c = vec3f(0.62, 0.5, 0.3); }
  return srgb_to_linear(c) * (0.9 + 0.1 * value_noise(p.xz * 140.0));
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4f {
  let obj = objects[v.id];
  let mat_id = u32(obj.material.y + 0.5);
  var n = normalize(v.normal);
  let vdir = normalize(frame.cameraPos - v.world_pos);

  var s = Surface(obj.color.rgb, obj.color.a, obj.material.x, vec3f(0.0));
  if (mat_id == MAT_ROOM) { s = room_surface(v.world_pos, n); }
  if (mat_id == MAT_RUG) { s.albedo = rug_surface(v.world_pos, obj.color.rgb); }
  if (mat_id == MAT_PAINTING) {
    // A dark landscape: sky, hills, a moon.
    let q = (obj.model[3].yz - v.world_pos.yz);
    let h = 0.05 * sin(q.y * 9.0) + 0.04 * sin(q.y * 23.0 + 1.0);
    s.albedo = select(vec3f(0.1, 0.16, 0.22), vec3f(0.12, 0.1, 0.05), q.x > h);
    s.albedo += vec3f(0.5, 0.45, 0.3) * smoothstep(0.045, 0.035, length(q - vec2f(-0.12, 0.1)));
  }
  if (mat_id == MAT_FLAME) {
    // The flame: emission only, hotter toward the core.
    let core = pow(saturate(dot(n, vdir)), 2.0);
    let col = frame.lightColor.rgb * frame.flameGlow * (0.4 + 1.6 * core);
    return vec4f(select(col, vec3f(1.0), frame.debugView != 0u), 1.0);
  }

  let to_light = frame.lightPos - v.world_pos;
  let dist2 = dot(to_light, to_light);
  let l = to_light * inverseSqrt(dist2);
  let ndl = dot(n, l);

  var vis = point_shadow(v.world_pos, n, saturate(ndl), v.position.xy);
  // Terminator fix: fade shadow lookups on surfaces turning away from the light.
  vis *= smoothstep(0.0, 0.1, ndl);

  // Inverse-square falloff, regularised by the flame's radius so it stays finite.
  let atten = 1.0 / (dist2 + frame.lightRadius * frame.lightRadius);
  let m = Material(s.albedo, s.roughness, s.metallic);
  // Ambient: cool moonlight from the window side, plus a warm one-bounce guess
  // proportional to the lantern's power (no GI here, just a believable floor).
  let hemi = mix(0.6, 1.0, n.y * 0.5 + 0.5);
  let ambient = (frame.ambientColor.rgb + frame.bounceColor.rgb) * hemi;
  var col = brdf_ambient(m, n, vdir, ambient, ambient * 0.5) + frame.lightColor.rgb * atten * vis * brdf_direct(m, n, vdir, l);
  col += s.emissive;

  if (frame.debugView == 1u) { col = vec3f(vis); }
  if (frame.debugView == 2u) {
    // Which cube face (or paraboloid half) each pixel reads from.
    let d = v.world_pos - frame.lightPos;
    let a = abs(d);
    var c = vec3f(0.0);
    if (frame.technique == TECH_CUBE) {
      if (a.x >= a.y && a.x >= a.z) { c = select(vec3f(0.5, 0.1, 0.1), vec3f(1.0, 0.25, 0.2), d.x > 0.0); }
      else if (a.y >= a.z) { c = select(vec3f(0.1, 0.4, 0.1), vec3f(0.3, 1.0, 0.3), d.y > 0.0); }
      else { c = select(vec3f(0.1, 0.15, 0.5), vec3f(0.3, 0.45, 1.0), d.z > 0.0); }
    } else {
      c = select(vec3f(0.2, 0.3, 0.9), vec3f(1.0, 0.6, 0.2), d.y >= 0.0);
    }
    col = c * (0.35 + 0.65 * vis);
  }
  if (frame.debugView == 3u) {
    // Linear distance to the light, banded every 0.5 m.
    let d = sqrt(dist2);
    col = vec3f(d / frame.shadowFar) * (0.85 + 0.15 * step(0.5, fract(d * 2.0)));
  }
  return vec4f(col, 1.0);
}

// --- Debug overlay: the shadow map, unfolded ----------------------------------

/// Direction for face `f` (+X, −X, +Y, −Y, +Z, −Z) at face coordinates sc, tc in [−1, 1]
/// (the cube-sampling convention: t runs down the face).
fn cube_dir(f: u32, sc: f32, tc: f32) -> vec3f {
  switch f {
    case 0u: { return vec3f(1.0, -tc, -sc); }
    case 1u: { return vec3f(-1.0, -tc, sc); }
    case 2u: { return vec3f(sc, 1.0, tc); }
    case 3u: { return vec3f(sc, -1.0, -tc); }
    case 4u: { return vec3f(sc, -tc, 1.0); }
    default: { return vec3f(-sc, -tc, -1.0); }
  }
}

fn show_distance(d: f32) -> vec3f {
  // Near = warm white, far = dark blue; bands every 0.5 m make seams obvious.
  let t = saturate(d / 6.0);
  let c = mix(vec3f(1.0, 0.85, 0.6), vec3f(0.05, 0.08, 0.2), sqrt(t));
  return c * (0.85 + 0.15 * step(0.5, fract(d * 2.0)));
}

@fragment
fn fs_overlay(v: FullscreenOut) -> @location(0) vec4f {
  let n = frame.shadowNear;
  let f = frame.shadowFar;
  if (frame.technique == TECH_CUBE) {
    // Horizontal cross: +Y above, then −X +Z +X −Z, then −Y.
    let cell = vec2u(v.uv * vec2f(4.0, 3.0));
    let fuv = fract(v.uv * vec2f(4.0, 3.0));
    var face_id = 6u;
    if (cell.y == 1u) { face_id = array<u32, 4>(1u, 4u, 0u, 5u)[cell.x]; }
    else if (cell.x == 1u) { face_id = select(3u, 2u, cell.y == 0u); }
    if (face_id == 6u) { discard; }
    let dir = cube_dir(face_id, fuv.x * 2.0 - 1.0, fuv.y * 2.0 - 1.0);
    let z = textureSampleLevel(shadow_cube, shadow_point, dir, 0);
    let ma = n / max(1.0 - z * (f - n) / f, 1e-5);
    var c = show_distance(ma * length(dir) / max_axis(dir));
    let border = min(min(fuv.x, 1.0 - fuv.x), min(fuv.y, 1.0 - fuv.y));
    c = mix(vec3f(0.9, 0.2, 0.2), c, step(0.006, border));
    return vec4f(c, 1.0);
  }
  // Two paraboloid discs side by side: upper hemisphere, lower hemisphere.
  let side_idx = u32(v.uv.x * 2.0);
  let uv = vec2f(fract(v.uv.x * 2.0), v.uv.y);
  let r = length(uv * 2.0 - 1.0);
  if (r > 1.0) { discard; }
  let z = textureSampleLevel(shadow_dp, shadow_point, uv, side_idx, 0);
  var c = show_distance(n + z * (f - n));
  c = mix(vec3f(0.9, 0.2, 0.2), c, step(r, 0.99));
  return vec4f(c, 1.0);
}
