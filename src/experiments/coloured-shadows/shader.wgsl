// ---------------------------------------------------------------------------
// Coloured Shadows — sunlight through stained glass and tinted acrylic.
//
// Two ways to shadow translucent casters, compared on one scene:
//
//   transmittance map (mode 0)
//     1. opaque casters -> depth map (layer 0 of the shadow array);
//     2. translucent casters -> an rgba16float "tint" map rendered from the
//        light, depth-tested (read-only) against that depth map, blended
//        rgb *= T (multiplicative: order-independent) and a = min(depth);
//     3. a receiver is lit by sun * opaque visibility * (T if it lies behind
//        the nearest translucent surface, else 1).
//   stochastic transparency (modes 1, 2)
//     translucent casters write depth like opaque ones, but each shadow-map
//     texel discards with probability T (hashed); filtering (and TAA, which
//     re-rolls the hash every frame) averages the holes into partial shadow.
//     Mode 1 uses one map and grey (luminance) opacity; mode 2 renders one
//     map per colour channel, so it can be coloured, for 3× the passes.
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, noise.wgsl, VertexIn,
// Object (props), brdf.wgsl, the `Frame` and `ShadowPass` structs.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var shadow_map: texture_depth_2d_array;
@group(0) @binding(3) var shadow_cmp: sampler_comparison;
@group(0) @binding(4) var tint_map: texture_2d<f32>;
@group(0) @binding(5) var shadow_cmp_point: sampler_comparison;

const STAINED_A: f32 = 1.0;
const STAINED_B: f32 = 2.0;
const ACRYLIC: f32 = 3.0;
const STONE: vec3f = vec3f(0.62, 0.56, 0.49);
const LEAD: vec3f = vec3f(0.05, 0.05, 0.055);

const POISSON = array<vec2f, 16>(
  vec2f(-0.94201624, -0.39906216), vec2f(0.94558609, -0.76890725),
  vec2f(-0.09418410, -0.92938870), vec2f(0.34495938, 0.29387760),
  vec2f(-0.91588581, 0.45771432), vec2f(-0.81544232, -0.87912464),
  vec2f(-0.38277543, 0.27676845), vec2f(0.97484398, 0.75648379),
  vec2f(0.44323325, -0.97511554), vec2f(0.53742981, -0.47373420),
  vec2f(-0.26496911, -0.41893023), vec2f(0.79197514, 0.19090188),
  vec2f(-0.24188840, 0.99706507), vec2f(-0.81409955, 0.91437590),
  vec2f(0.19984126, 0.78641367), vec2f(0.14383161, -0.14100790),
);

// --- Glass ---------------------------------------------------------------------------

struct Glass {
  /// Linear transmittance per channel (0 where solid).
  t: vec3f,
  /// 1 where the pane is solid: lead cames, or stone outside the arch.
  solid: f32,
  /// Colour of the solid part.
  solid_color: vec3f,
};

fn palette(h: f32) -> vec3f {
  // Pot-metal glass: ruby, cobalt, amber, emerald, violet and a pale "white" glass.
  if (h < 0.2) { return vec3f(0.85, 0.06, 0.08); }
  if (h < 0.4) { return vec3f(0.08, 0.2, 0.85); }
  if (h < 0.58) { return vec3f(0.95, 0.58, 0.06); }
  if (h < 0.72) { return vec3f(0.12, 0.66, 0.22); }
  if (h < 0.82) { return vec3f(0.45, 0.12, 0.7); }
  return vec3f(0.92, 0.88, 0.72);
}

/// A lancet window: pointed arch, a diamond-quarry field and a round medallion.
/// `p` is in pane units (x in [-0.6, 0.6], y in [0, 2.4]).
fn stained_glass(p: vec2f, variant: f32) -> Glass {
  var g: Glass;
  g.t = vec3f(0.0);
  g.solid = 1.0;
  g.solid_color = STONE;
  // Pointed (equilateral) arch springing at y = 1.36; outside it is stone tracery.
  let spring = 1.36;
  let inside_arch = p.y < spring || (distance(p, vec2f(0.6, spring)) < 1.2 && distance(p, vec2f(-0.6, spring)) < 1.2);
  let edge = min(min(0.6 - abs(p.x), p.y), select(1.2 - max(distance(p, vec2f(0.6, spring)), distance(p, vec2f(-0.6, spring))), 1.0, p.y < spring));
  if (!inside_arch || edge < 0.035) { return g; }
  g.solid_color = LEAD;
  let lead = 0.011;

  // Medallion: a ring of petals around a centre boss.
  let mc = vec2f(0.0, 1.62);
  let r = distance(p, mc);
  var cell_hash: f32;
  if (r < 0.4) {
    if (abs(r - 0.4) < lead * 1.5 || abs(r - 0.13) < lead) { return g; }
    if (r < 0.13) {
      cell_hash = fract(variant * 0.37 + 0.55);
    } else {
      let a = atan2(p.y - mc.y, p.x - mc.x) / TAU + 0.5;
      let sector = floor(a * 8.0);
      let along = fract(a * 8.0);
      if (min(along, 1.0 - along) * r * TAU / 8.0 < lead) { return g; }
      cell_hash = select(fract(variant * 0.61 + 0.05), fract(variant * 0.29 + 0.31), (sector % 2.0) == 0.0);
    }
  } else {
    // Diamond quarries: a 45°-rotated grid.
    let q = vec2f(p.x + p.y, p.y - p.x) * 0.70710678 / 0.2;
    let cell = floor(q);
    let f = fract(q);
    if (min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)) * 0.2 < lead) { return g; }
    cell_hash = hash21(cell + variant * 17.0);
    // A border band of amber glass inside the frame.
    if (edge < 0.1) {
      if (edge < 0.035 + lead) { return g; }
      cell_hash = 0.5;
    }
  }
  var t = palette(cell_hash);
  // Antique glass is uneven: seedy, streaky density.
  t *= 0.82 + 0.18 * value_noise(p * vec2f(9.0, 23.0) + variant * 5.0);
  g.t = t;
  g.solid = 0.0;
  return g;
}

/// Transmittance of a translucent prop at object-space position `local` (unit box).
fn glass_at(obj: Object, local: vec3f) -> Glass {
  let kind = obj.material.y;
  if (kind == ACRYLIC) {
    var g: Glass;
    g.t = obj.color.rgb;
    g.solid = 0.0;
    g.solid_color = vec3f(0.0);
    return g;
  }
  // Stained-glass panes are 1.2 × 2.4 world units.
  return stained_glass(vec2f(local.x * 1.2, (local.y + 0.5) * 2.4), kind);
}

// --- Light-space passes --------------------------------------------------------------

@vertex
fn vs_shadow(v: VertexIn, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return frame.lightViewProj * objects[id].model * vec4f(v.position, 1.0);
}

struct TransmitOut {
  @builtin(position) position: vec4f,
  @location(0) local: vec3f,
  @location(1) @interpolate(flat) id: u32,
};

@vertex
fn vs_transmit(v: VertexIn, @builtin(instance_index) id: u32) -> TransmitOut {
  var out: TransmitOut;
  out.position = frame.lightViewProj * objects[id].model * vec4f(v.position, 1.0);
  out.local = v.position;
  out.id = id;
  return out;
}

/// Transmittance map: rgb is multiplied into the target, alpha keeps the nearest depth (min blend).
@fragment
fn fs_transmit(v: TransmitOut) -> @location(0) vec4f {
  let g = glass_at(objects[v.id], v.local);
  return vec4f(g.t, v.position.z);
}

@group(1) @binding(0) var<uniform> shadow_pass: ShadowPass;

/// Stochastic transparency: keep this texel's depth with probability (1 - T).
@fragment
fn fs_stochastic(v: TransmitOut) {
  let g = glass_at(objects[v.id], v.local);
  var t = luminance(g.t);
  if (shadow_pass.channel < 3u) { t = g.t[shadow_pass.channel]; }
  // A per-texel hash, re-rolled every frame when TAA can average it.
  let h = hash12(floor(v.position.xy) + shadow_pass.seed);
  if (h < t) { discard; }
}

// --- Receivers -----------------------------------------------------------------------

fn to_shadow_uv(p: vec3f) -> vec3f {
  let c = frame.lightViewProj * vec4f(p, 1.0);
  let ndc = c.xyz / c.w;
  return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

/// Sun light reaching `p`, per channel: opaque visibility × translucent tint.
fn sun_filter(p: vec3f, n: vec3f, ndl: f32, frag: vec2f) -> vec3f {
  let offset = n * frame.normalOffset * frame.shadowTexelWorld * (1.0 - 0.5 * ndl);
  var lp = to_shadow_uv(p + offset);
  lp.z -= frame.depthBias;
  if (any(lp.xy <= vec2f(0.0)) || any(lp.xy >= vec2f(1.0)) || lp.z >= 1.0) { return vec3f(1.0); }
  let dims = vec2f(textureDimensions(tint_map));
  let texel = 1.0 / dims.x;
  let angle = ign(frag + frame.noiseOffset) * TAU;
  let rot = mat2x2f(cos(angle), sin(angle), -sin(angle), cos(angle));
  var disk = POISSON;
  let radius = frame.filterRadius * texel;
  var sum = vec3f(0.0);
  for (var i = 0; i < 16; i++) {
    let uv = lp.xy + rot * disk[i] * radius;
    if (frame.mode == 0u) {
      // Only receivers *behind* the nearest translucent surface are tinted. Both lookups are
      // point-sampled, so each tap judges one texel consistently: a bilinear depth would blend
      // "no glass" (1.0) into the glass depth at silhouettes, and a bilinear opaque test next to a
      // nearest tint lets half-lit, untinted light leak along every edge where the wall hides glass.
      let vis = textureSampleCompareLevel(shadow_map, shadow_cmp_point, uv, 0, lp.z);
      let tint = textureLoad(tint_map, clamp(vec2i(uv * dims), vec2i(0), vec2i(dims) - 1), 0);
      sum += vis * select(vec3f(1.0), tint.rgb, lp.z > tint.a + frame.tintBias);
    } else if (frame.mode == 1u) {
      sum += vec3f(textureSampleCompareLevel(shadow_map, shadow_cmp, uv, 0, lp.z));
    } else {
      sum += vec3f(
        textureSampleCompareLevel(shadow_map, shadow_cmp, uv, 0, lp.z),
        textureSampleCompareLevel(shadow_map, shadow_cmp, uv, 1, lp.z),
        textureSampleCompareLevel(shadow_map, shadow_cmp, uv, 2, lp.z));
    }
  }
  return sum / 16.0;
}

fn sky_radiance(dir: vec3f) -> vec3f {
  let h = saturate(dir.y);
  return mix(frame.horizonColor, frame.zenithColor, sqrt(h));
}

fn ambient(n: vec3f) -> vec3f {
  // Sky from above, the sunlit floor bouncing a little light from below.
  return mix(frame.horizonColor * 0.35, mix(frame.horizonColor, frame.zenithColor, 0.75), n.y * 0.5 + 0.5) * frame.ambient;
}

struct SceneOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) local: vec3f,
  @location(3) @interpolate(flat) id: u32,
};

@vertex
fn vs_scene(v: VertexIn, @builtin(instance_index) id: u32) -> SceneOut {
  let m = objects[id].model;
  let wp = m * vec4f(v.position, 1.0);
  var out: SceneOut;
  out.position = frame.viewProj * wp;
  out.world_pos = wp.xyz;
  // Panes are axis-scaled unit boxes; normalising m·n is exact for their axis-aligned faces.
  out.normal = (m * vec4f(v.normal, 0.0)).xyz;
  out.local = v.position;
  out.id = id;
  return out;
}

fn fog(col: vec3f, p: vec3f) -> vec3f {
  let d = distance(p, frame.cameraPos);
  return mix(col, frame.horizonColor, smoothstep(frame.fogStart, frame.fogEnd, d));
}

@fragment
fn fs_opaque(v: SceneOut) -> @location(0) vec4f {
  let obj = objects[v.id];
  let n = normalize(v.normal);
  let l = frame.toLight;
  let vdir = normalize(frame.cameraPos - v.world_pos);
  let ndl = saturate(dot(n, l));
  let sun = sun_filter(v.world_pos, n, ndl, v.position.xy) * smoothstep(0.0, 0.08, ndl);
  let m = Material(obj.color.rgb, obj.color.a, obj.material.x);
  let r = reflect(-vdir, n);
  var col = brdf_ambient(m, n, vdir, ambient(n), sky_radiance(r) * frame.ambient) + frame.sunColor * sun * brdf_direct(m, n, vdir, l);
  if (frame.debugView == 1u) { col = sun * 0.85 + 0.05; }
  return vec4f(fog(col, v.world_pos), 1.0);
}

// Glass seen by the camera, in two blended draws so it filters *and* glows without
// dual-source blending: first dst *= transmittance, then dst += reflection + emission.

@fragment
fn fs_glass_filter(v: SceneOut) -> @location(0) vec4f {
  let g = glass_at(objects[v.id], v.local);
  let n = normalize(v.normal);
  let vdir = normalize(frame.cameraPos - v.world_pos);
  let f = 0.04 + 0.96 * pow(1.0 - abs(dot(n, vdir)), 5.0);
  return vec4f(g.t * (1.0 - f), 1.0);
}

@fragment
fn fs_glass_light(v: SceneOut) -> @location(0) vec4f {
  let obj = objects[v.id];
  let g = glass_at(obj, v.local);
  var n = normalize(v.normal);
  let vdir = normalize(frame.cameraPos - v.world_pos);
  if (dot(n, vdir) < 0.0) { n = -n; }
  let l = frame.toLight;
  let ndl = dot(n, l);
  let sun = sun_filter(v.world_pos, n * sign(ndl + 1e-4), abs(ndl), v.position.xy);
  // Solid parts (lead, stone tracery) are ordinary opaque surfaces: the filter pass zeroed them.
  let solid = Material(g.solid_color, 0.7, 0.0);
  let solid_col = brdf_ambient(solid, n, vdir, ambient(n), sky_radiance(reflect(-vdir, n)) * frame.ambient)
    + frame.sunColor * sun * brdf_direct(solid, n, vdir, l);
  // Glass: Fresnel reflection of the sky and sun, plus a little light scattered forward through
  // the (slightly diffusing) glass when it is lit from behind: why stained glass glows.
  let f = 0.04 + 0.96 * pow(1.0 - abs(dot(n, vdir)), 5.0);
  let r = reflect(-vdir, n);
  let spec = sky_radiance(r) * frame.ambient * f + frame.sunColor * sun * pow(saturate(dot(r, l)), 400.0) * 4.0;
  let backlit = saturate(-ndl);
  let glow = frame.sunColor * sun * g.t * backlit * frame.glassGlow * select(1.0, 0.35, obj.material.y == ACRYLIC);
  var col = mix(spec + glow, solid_col, g.solid);
  if (frame.debugView == 1u) { col = mix(vec3f(0.0), vec3f(0.3), g.solid); }
  return vec4f(fog(col, v.world_pos), 0.0);
}

// --- Sky -----------------------------------------------------------------------------

@fragment
fn fs_sky(v: FullscreenOut) -> @location(0) vec4f {
  let ndc = vec2f(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0);
  let far = frame.invViewProj * vec4f(ndc, 1.0, 1.0);
  let dir = normalize(far.xyz / far.w - frame.cameraPos);
  var col = sky_radiance(dir);
  // The sun disc and a soft glow around it.
  let c = dot(dir, frame.toLight);
  col += frame.sunColor * (smoothstep(0.9995, 0.9998, c) * 8.0 + pow(saturate(c), 64.0) * 0.12);
  if (frame.debugView == 1u) { col = vec3f(0.0); }
  return vec4f(col, 1.0);
}

// --- Debug overlay: one of the light-space maps in a corner ------------------------------

@fragment
fn fs_overlay(v: FullscreenOut) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(tint_map));
  let px = vec2i(v.uv * dims);
  if (frame.overlay == 1u) { return vec4f(textureLoad(tint_map, px, 0).rgb, 1.0); }
  if (frame.overlay == 2u) {
    let d = textureLoad(tint_map, px, 0).a;
    return vec4f(select(vec3f(pow(saturate(d), 4.0)), vec3f(0.12, 0.1, 0.2), d >= 1.0), 1.0);
  }
  let sdims = vec2f(textureDimensions(shadow_map));
  let d = textureLoad(shadow_map, vec2i(v.uv * sdims), 0, 0);
  return vec4f(vec3f(pow(saturate(d), 4.0)), 1.0);
}
