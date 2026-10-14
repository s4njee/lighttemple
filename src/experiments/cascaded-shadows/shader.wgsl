// ---------------------------------------------------------------------------
// Cascaded shadow maps over a large terrain.
// Outputs premultiplied linear HDR (alpha = coverage) into the post chain: the
// horizon haze and the far terrain dissolve into the page colour.
// Requires (prepended): common.wgsl, noise.wgsl, fullscreen.wgsl, VertexIn,
// PROP_WGSL (`Object`), the `Frame` and `ShadowPass` structs, brdf.wgsl.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var shadow_maps: texture_depth_2d_array;
@group(0) @binding(3) var shadow_cmp: sampler_comparison;
// Only the shadow pass binds this (one buffer per cascade).
@group(0) @binding(4) var<uniform> shadow_pass: ShadowPass;

const MAT_TERRAIN = 0u;
const MAT_FOLIAGE = 1u;
const MAT_ROCK = 2u;

// Same disc as shadow.wgsl, which binds a plain texture_depth_2d and so can't be reused with an array.
const CSM_POISSON = array<vec2f, 16>(
  vec2f(-0.94201624, -0.39906216), vec2f(0.94558609, -0.76890725),
  vec2f(-0.09418410, -0.92938870), vec2f(0.34495938, 0.29387760),
  vec2f(-0.91588581, 0.45771432), vec2f(-0.81544232, -0.87912464),
  vec2f(-0.38277543, 0.27676845), vec2f(0.97484398, 0.75648379),
  vec2f(0.44323325, -0.97511554), vec2f(0.53742981, -0.47373420),
  vec2f(-0.26496911, -0.41893023), vec2f(0.79197514, 0.19090188),
  vec2f(-0.24188840, 0.99706507), vec2f(-0.81409955, 0.91437590),
  vec2f(0.19984126, 0.78641367), vec2f(0.14383161, -0.14100790),
);

fn cascade_matrix(c: u32) -> mat4x4f {
  switch c {
    case 0u: { return frame.cascade0; }
    case 1u: { return frame.cascade1; }
    case 2u: { return frame.cascade2; }
    default: { return frame.cascade3; }
  }
}

/// Model-matrix normal transform that survives non-uniform scale: the cofactor matrix
/// (the inverse-transpose up to a scale factor, which the normalize removes).
fn transform_normal(m: mat4x4f, n: vec3f) -> vec3f {
  let a = m[0].xyz; let b = m[1].xyz; let c = m[2].xyz;
  return normalize(mat3x3f(cross(b, c), cross(c, a), cross(a, b)) * n);
}

// --- Shadow (depth-only) pass: one per cascade, each into its own array layer --

@vertex
fn vs_shadow(v: VertexIn, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return shadow_pass.viewProj * objects[id].model * vec4f(v.position, 1.0);
}

// --- Array-aware filters (cf. shadow.wgsl) -------------------------------------

fn csm_hard(lp: vec3f, layer: u32) -> f32 {
  return textureSampleCompareLevel(shadow_maps, shadow_cmp, lp.xy, layer, lp.z);
}

fn csm_pcf(lp: vec3f, layer: u32, texel: f32, r: i32) -> f32 {
  var sum = 0.0;
  for (var y = -r; y <= r; y++) {
    for (var x = -r; x <= r; x++) {
      sum += textureSampleCompareLevel(shadow_maps, shadow_cmp, lp.xy + vec2f(f32(x), f32(y)) * texel, layer, lp.z);
    }
  }
  let n = f32(2 * r + 1);
  return sum / (n * n);
}

fn csm_poisson(lp: vec3f, layer: u32, radius_uv: f32, angle: f32) -> f32 {
  let c = cos(angle); let s = sin(angle);
  let rot = mat2x2f(c, s, -s, c);
  var disk = CSM_POISSON;
  var sum = 0.0;
  for (var i = 0; i < 16; i++) {
    sum += textureSampleCompareLevel(shadow_maps, shadow_cmp, lp.xy + rot * disk[i] * radius_uv, layer, lp.z);
  }
  return sum / 16.0;
}

// --- Cascade selection & sampling ------------------------------------------------

/// Radius (in texels, plus one for the bilinear footprint) that the current filter reaches.
fn filter_reach(c: u32) -> f32 {
  switch frame.filterMode {
    case 0u: { return 1.0; }
    case 1u: { return 2.0; }
    case 2u: { return 3.0; }
    default: { return max(frame.softness / frame.texelWorld[c], 1.0) + 1.0; }
  }
}

/// Light-space lookup (uv, depth) into cascade `c`. Both biases are measured in that cascade's
/// texels, so they scale with its resolution: a fixed world-space bias would be far too small for
/// the far cascades (acne) or far too big for the near one (peter-panning). The depth bias is also
/// slope-scaled and grows with the filter's reach: a wider kernel compares against texels further
/// away, and on a surface lit at a grazing angle those are proportionally deeper.
fn shadow_coord(c: u32, wp: vec3f, n: vec3f, ndl: f32) -> vec3f {
  let texel = frame.texelWorld[c];
  let offset = n * frame.normalOffset * texel;
  let tan_theta = min(sqrt(1.0 - ndl * ndl) / max(ndl, 1e-3), 5.0);
  let bias = frame.depthBias * texel * filter_reach(c) * (0.5 + tan_theta);
  let clip = cascade_matrix(c) * vec4f(wp + offset, 1.0);
  return vec3f(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5, clip.z - bias / frame.depthRange[c]);
}

fn sample_cascade(c: u32, wp: vec3f, n: vec3f, ndl: f32, angle: f32) -> f32 {
  if (c >= frame.cascadeCount) { return 1.0; }
  let lp = shadow_coord(c, wp, n, ndl);
  if (any(lp.xy < vec2f(0.0)) || any(lp.xy > vec2f(1.0))) { return 1.0; }
  let texel_uv = 1.0 / frame.mapSize;
  switch frame.filterMode {
    case 0u: { return csm_hard(lp, c); }
    case 1u: { return csm_pcf(lp, c, texel_uv, 1); }
    case 2u: { return csm_pcf(lp, c, texel_uv, 2); }
    default: {
      // A world-space radius gives the same penumbra in every cascade (but never under one texel).
      let radius = max(frame.softness / (frame.texelWorld[c] * frame.mapSize), texel_uv);
      return csm_poisson(lp, c, radius, angle);
    }
  }
}

struct CascadePick {
  /// Cascade index (cascadeCount = beyond the shadow distance).
  index: u32,
  /// 0..1 progress through the blend band at the far end of `index`.
  blend: f32,
};

fn pick_cascade(wp: vec3f) -> CascadePick {
  // Distance along the view axis, not to the eye, so splits are planes parallel to the screen.
  let depth = dot(wp - frame.cascadeOrigin, frame.cascadeAxis);
  var pick = CascadePick(frame.cascadeCount, 0.0);
  for (var i = 0u; i < frame.cascadeCount; i++) {
    if (depth < frame.splits[i]) {
      pick.index = i;
      let band = frame.bands[i];
      if (band > 0.0) { pick.blend = saturate((depth - (frame.splits[i] - band)) / band); }
      break;
    }
  }
  return pick;
}

/// Shadow visibility with cascade blending. Returns (visibility, cascade, blend) so the debug views
/// can show which cascade(s) were used.
fn shadow_visibility(wp: vec3f, n: vec3f, ndl: f32, frag: vec2f) -> vec3f {
  var pick = pick_cascade(wp);
  if (frame.blendMode == 1u) {
    // Dither: each pixel takes one cascade or the next, with the probability rising across the band.
    // One lookup instead of two; the noise is what TAA (or grain) has to hide.
    if (pick.blend > ign(frag + frame.noiseOffset * 1.37 + 19.0)) { pick.index += 1u; }
    pick.blend = 0.0;
  } else if (frame.blendMode == 2u) {
    pick.blend = 0.0;
  }
  let angle = ign(frag + frame.noiseOffset) * TAU;
  var vis = sample_cascade(pick.index, wp, n, ndl, angle);
  // Lerp: in the band, sample both cascades. Past the last cascade "the next one" is fully lit, so
  // shadows fade out at the shadow distance instead of stopping at a line.
  if (pick.blend > 0.0) {
    vis = mix(vis, sample_cascade(pick.index + 1u, wp, n, ndl, angle), pick.blend);
  }
  return vec3f(vis, f32(pick.index), pick.blend);
}

/// Debug colour per cascade; grey beyond the last one.
fn cascade_tint(c: u32) -> vec3f {
  if (c >= frame.cascadeCount) { return vec3f(0.6); }
  switch c {
    case 0u: { return vec3f(1.0, 0.22, 0.18); }
    case 1u: { return vec3f(0.25, 0.95, 0.3); }
    case 2u: { return vec3f(0.25, 0.5, 1.0); }
    case 3u: { return vec3f(1.0, 0.85, 0.15); }
    default: { return vec3f(0.6); }
  }
}

// --- Atmosphere -------------------------------------------------------------------

/// Haze colour along a view ray, warmer toward the sun.
fn haze(ray: vec3f) -> vec3f {
  let toward_sun = saturate(dot(ray, frame.toSun));
  return frame.hazeColor.rgb + frame.sunColor.rgb * (0.05 * pow(toward_sun, 8.0) + 0.04 * pow(toward_sun, 64.0));
}

/// Coverage of the haze band above the horizon; above it the page colour shows through.
fn sky_alpha(ray: vec3f) -> f32 {
  return frame.hazeColor.a * exp(-max(ray.y, 0.0) * 6.0);
}

// --- Main pass ------------------------------------------------------------------------

struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) @interpolate(flat) id: u32,
};

@vertex
fn vs_main(v: VertexIn, @builtin(instance_index) id: u32) -> VsOut {
  let m = objects[id].model;
  let wp = m * vec4f(v.position, 1.0);
  var out: VsOut;
  out.position = frame.viewProj * wp;
  out.world_pos = wp.xyz;
  out.normal = transform_normal(m, v.normal);
  out.uv = v.uv;
  out.id = id;
  return out;
}

fn terrain_albedo(p: vec3f, n: vec3f, masks: vec2f) -> vec3f {
  // Grass in patches, drier swathes, darker forest floor, the dirt track, rock on steep ground and
  // pale stone on the peaks.
  var c = frame.grassColor.rgb * mix(0.72, 1.15, value_noise(p.xz * 0.06)) * mix(0.9, 1.06, value_noise(p.xz * 0.5));
  c = mix(c, vec3f(0.36, 0.3, 0.12), smoothstep(0.55, 0.85, value_noise(p.xz * 0.011 + 7.0)) * 0.55);
  c = mix(c, vec3f(0.06, 0.06, 0.025), masks.y * 0.75);
  c = mix(c, vec3f(0.33, 0.24, 0.15) * mix(0.85, 1.1, value_noise(p.xz * 1.3)), saturate(masks.x));
  let steep = smoothstep(0.86, 0.68, n.y);
  c = mix(c, vec3f(0.23, 0.21, 0.18) * mix(0.8, 1.15, value_noise(p.xz * 0.2)), steep);
  let peak = smoothstep(120.0, 170.0, p.y + 25.0 * value_noise(p.xz * 0.02)) * smoothstep(0.6, 0.8, n.y);
  return mix(c, vec3f(0.62, 0.6, 0.57), peak);
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4f {
  let obj = objects[v.id];
  let material = u32(obj.material.y + 0.5);
  let n = normalize(v.normal);
  let l = frame.toSun;
  let to_eye = frame.cameraPos - v.world_pos;
  let dist = length(to_eye);
  let vdir = to_eye / dist;
  let ndl = saturate(dot(n, l));

  var albedo = obj.color.rgb;
  var ao = 1.0;
  if (material == MAT_TERRAIN) {
    albedo = terrain_albedo(v.world_pos, n, v.uv);
  } else {
    // uv.x flags bark, uv.y is relative height in the mesh: a free ambient-occlusion gradient.
    if (v.uv.x > 0.5) { albedo = vec3f(0.07, 0.045, 0.03); }
    ao = mix(select(0.6, 0.35, material == MAT_FOLIAGE), 1.0, saturate(v.uv.y));
  }

  let s = shadow_visibility(v.world_pos, n, ndl, v.position.xy);
  // Terminator fix: surfaces turning away from the sun are already dark from N·L.
  let vis = s.x * smoothstep(0.0, 0.06, ndl);

  let m = Material(albedo, obj.color.a, 0.0);
  let irradiance = mix(frame.groundColor.rgb, frame.skyColor.rgb, n.y * 0.5 + 0.5) * ao;
  let r = reflect(-vdir, n);
  let env = mix(frame.groundColor.rgb, frame.skyColor.rgb, smoothstep(-0.6, 0.6, r.y)) * ao;
  var col = brdf_ambient(m, n, vdir, irradiance, env) + frame.sunColor.rgb * vis * brdf_direct(m, n, vdir, l);

  let cascade = u32(s.y + 0.5);
  if (frame.debugView == 1u) {
    let tint = mix(cascade_tint(cascade), cascade_tint(cascade + 1u), s.z);
    col = tint * (0.08 + 0.6 * luminance(col));
  } else if (frame.debugView == 2u) {
    col = vec3f(s.x);
  } else if (frame.debugView == 3u) {
    // Shadow-map texels as a checkerboard: how much resolution each part of the screen gets.
    var checker = 0.5;
    if (cascade < frame.cascadeCount) {
      let t = floor(shadow_coord(cascade, v.world_pos, n, ndl).xy * frame.mapSize);
      checker = select(0.55, 1.0, (i32(t.x) + i32(t.y)) % 2 == 0);
    }
    col = cascade_tint(cascade) * checker * (0.25 + 0.75 * s.x) * 0.5;
  }

  // Aerial perspective, then the far terrain dissolves into the sky band (premultiplied lerp), which
  // itself fades into the page colour after tonemapping.
  let ray = -vdir;
  if (frame.debugView == 0u) {
    col = mix(haze(ray), col, exp(-dist * frame.fogDensity));
  }
  let fade = smoothstep(frame.fadeStart, frame.fadeEnd, dist);
  let sky_a = sky_alpha(ray);
  return vec4f(mix(col, haze(ray) * sky_a, fade), mix(1.0, sky_a, fade));
}

// --- Sky: a haze band over the page colour, plus the sun --------------------------------

@fragment
fn fs_sky(v: FullscreenOut) -> @location(0) vec4f {
  let ndc = vec2f(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0);
  let far = frame.invViewProj * vec4f(ndc, 1.0, 1.0);
  let ray = normalize(far.xyz / far.w - frame.cameraPos);
  let a = sky_alpha(ray);
  let sun = smoothstep(0.99965, 0.9998, dot(ray, frame.toSun));
  return vec4f(haze(ray) * a + frame.sunColor.rgb * 3.0 * sun, max(a, sun));
}

// --- Debug: the cascade maps as thumbnails ------------------------------------------

struct ThumbOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) layer: u32,
};

@vertex
fn vs_thumb(@builtin(vertex_index) i: u32, @builtin(instance_index) layer: u32) -> ThumbOut {
  // The fullscreen triangle (cf. fullscreen.wgsl); the viewport places it.
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return ThumbOut(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y), layer);
}

@fragment
fn fs_thumb(v: ThumbOut) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(shadow_maps));
  let px = clamp(vec2i(v.uv * dims), vec2i(0), vec2i(dims) - 1);
  let d = textureLoad(shadow_maps, px, v.layer, 0);
  // Stretch the receivers' depth range (plus a little caster padding) to 0..1: the padding that
  // reaches back toward the sun is mostly empty.
  let rs = frame.receiverStart[v.layer];
  let lo = max(0.0, rs - 0.5 * (1.0 - rs));
  let x = saturate((d - lo) / (1.0 - lo));
  var c = vec3f(pow(1.0 - x, 1.5));
  let edge = min(min(v.uv.x, v.uv.y), min(1.0 - v.uv.x, 1.0 - v.uv.y));
  c = select(c, cascade_tint(v.layer), edge < 0.02);
  return vec4f(c, 1.0);
}
