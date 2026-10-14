// ---------------------------------------------------------------------------
// Spotlight & Gobo: a perspective shadow map for a spot light, plus a
// projected cookie texture (procedural, regenerated when it animates).
// Outputs premultiplied linear HDR (alpha = coverage) into the post chain.
// Requires (prepended): common.wgsl, fullscreen.wgsl, noise.wgsl, brdf.wgsl,
// VertexIn, the `Object` struct, and the `Frame` and `Cookie` structs.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(3) var shadow_cmp: sampler_comparison;
@group(0) @binding(4) var cookie_tex: texture_2d<f32>;
@group(0) @binding(5) var cookie_smp: sampler;

const MAT_PLAIN = 0u;
const MAT_WALL = 1u;
const MAT_FLOOR = 2u;

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

// --- Shadow (depth-only) pass ----------------------------------------------

@vertex
fn vs_shadow(v: VertexIn, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return frame.lightViewProj * objects[id].model * vec4f(v.position, 1.0);
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
  out.normal = normalize((m * vec4f(v.normal, 0.0)).xyz);
  out.id = id;
  return out;
}

/// Non-linear depth the perspective map stores for a point `z` metres in front of the light.
fn ndc_depth(z: f32) -> f32 {
  let n = frame.shadowNear;
  let f = frame.shadowFar;
  return f / (f - n) * (1.0 - n / z);
}

fn linear_depth(d: f32) -> f32 {
  let n = frame.shadowNear;
  let f = frame.shadowFar;
  return n / max(1.0 - d * (f - n) / f, 1e-6);
}

struct LightSpace {
  uv: vec2f,
  // Distance along the light's axis (linear view depth), metres.
  z: f32,
  // World size of one shadow texel at that depth.
  texel_world: f32,
};

fn light_space(p: vec3f) -> LightSpace {
  let clip = frame.lightViewProj * vec4f(p, 1.0);
  var ls: LightSpace;
  ls.uv = vec2f(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
  ls.z = clip.w;
  // A perspective map's texels grow linearly with distance: the root of perspective aliasing.
  ls.texel_world = 2.0 * ls.z * frame.tanHalfFov / frame.mapSize;
  return ls;
}

fn spot_shadow(world_pos: vec3f, n: vec3f, ndl: f32, frag: vec2f) -> f32 {
  if (frame.filterMode == 0u) { return 1.0; }
  let ls0 = light_space(world_pos);
  // Normal offset scales with the texel footprint, so it is constant in texels
  // whether the receiver is near the light or far away.
  let p = world_pos + n * frame.normalOffset * ls0.texel_world * (1.5 - ndl);
  let ls = light_space(p);
  if (any(ls.uv < vec2f(0.0)) || any(ls.uv > vec2f(1.0)) || ls.z <= frame.shadowNear) { return 1.0; }
  // Bias is applied in *linear* depth, then converted: a constant NDC bias would
  // be enormous near the light and vanish far away (depth is ~1/z).
  let ref_depth = ndc_depth(ls.z - frame.depthBias * ls.texel_world);
  let texel = 1.0 / frame.mapSize;
  let angle = ign(frag + frame.noiseOffset) * TAU;
  let rot = mat2x2f(cos(angle), sin(angle), -sin(angle), cos(angle));
  var disk = POISSON_16;

  if (frame.filterMode == 1u) { return textureSampleCompareLevel(shadow_map, shadow_cmp, ls.uv, ref_depth); }

  var radius = frame.pcfRadius * texel;
  if (frame.filterMode == 3u) {
    // PCSS with a perspective light: the light (radius R) is a disc at the apex.
    // uv units at depth z span 2·z·tan(fov/2) metres.
    let uv_per_m_near = 1.0 / (2.0 * frame.shadowNear * frame.tanHalfFov);
    let search = clamp(frame.lightSize * (ls.z - frame.shadowNear) / ls.z * uv_per_m_near, 2.0 * texel, 40.0 * texel);
    let dims = vec2f(textureDimensions(shadow_map));
    var sum = 0.0;
    var count = 0.0;
    for (var i = 0; i < 16; i++) {
      let uv = ls.uv + rot * disk[i] * search;
      let px = clamp(vec2i(uv * dims), vec2i(0), vec2i(dims) - 1);
      let zb = linear_depth(textureLoad(shadow_map, px, 0));
      if (zb < ls.z - frame.depthBias * ls.texel_world) {
        sum += zb;
        count += 1.0;
      }
    }
    if (count == 0.0) { return 1.0; }
    let zb = sum / count;
    // Penumbra R·(zR − zB)/zB metres at the receiver, in uv at the receiver's depth.
    let penumbra_uv = frame.lightSize * (ls.z - zb) / zb / (2.0 * ls.z * frame.tanHalfFov);
    radius = clamp(penumbra_uv, texel, 40.0 * texel);
  }
  var vis = 0.0;
  for (var i = 0; i < 16; i++) {
    vis += textureSampleCompareLevel(shadow_map, shadow_cmp, ls.uv + rot * disk[i] * radius, ref_depth);
  }
  return vis / 16.0;
}

/// The light's colour pattern at `p`: cone falloff × cookie.
fn spot_pattern(p: vec3f) -> vec3f {
  let l = normalize(p - frame.lightPos);
  let cos_a = dot(l, frame.lightDir);
  let cone = smoothstep(frame.cosOuter, frame.cosInner, cos_a);
  if (cone <= 0.0) { return vec3f(0.0); }
  let clip = frame.lightViewProj * vec4f(p, 1.0);
  // The cookie is projected over the cone's square: uv from the light's own projection.
  let uv = (vec2f(clip.x, -clip.y) / clip.w) * frame.cookieScale * 0.5 + 0.5;
  var cookie = vec3f(1.0);
  if (frame.cookieOn != 0u) {
    cookie = textureSampleLevel(cookie_tex, cookie_smp, uv, frame.cookieBlur).rgb;
  }
  return cookie * cone;
}

fn wall_albedo(p: vec3f) -> vec3f {
  // Limewash: low-frequency mottling plus fine trowel grain.
  let m = fbm(p.xy * 0.6, 4) * 0.5 + 0.5;
  let g = fbm(p.xy * 9.0 + 3.0, 3);
  return frame.wallColor.rgb * (0.86 + 0.18 * m + 0.05 * g);
}

fn floor_albedo(p: vec3f) -> vec3f {
  // Polished concrete with faint aggregate.
  let m = fbm(p.xz * 0.8, 4) * 0.5 + 0.5;
  let speck = step(0.93, value_noise(p.xz * 60.0)) * 0.08;
  return frame.floorColor.rgb * (0.8 + 0.3 * m) + speck;
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4f {
  let obj = objects[v.id];
  let mat_id = u32(obj.material.y + 0.5);
  let n = normalize(v.normal);
  let vdir = normalize(frame.cameraPos - v.world_pos);

  var albedo = obj.color.rgb;
  var roughness = obj.color.a;
  if (mat_id == MAT_WALL) { albedo = wall_albedo(v.world_pos); }
  if (mat_id == MAT_FLOOR) {
    albedo = floor_albedo(v.world_pos);
    roughness = 0.32 + 0.2 * value_noise(v.world_pos.xz * 3.0);
  }

  let to_light = frame.lightPos - v.world_pos;
  let dist2 = dot(to_light, to_light);
  let l = to_light * inverseSqrt(dist2);
  let ndl = dot(n, l);
  let pattern = spot_pattern(v.world_pos);
  var vis = 0.0;
  if (max(pattern.r, max(pattern.g, pattern.b)) > 0.0) {
    vis = spot_shadow(v.world_pos, n, saturate(ndl), v.position.xy) * smoothstep(0.0, 0.1, ndl);
  }

  let m = Material(albedo, roughness, obj.material.x);
  let hemi = mix(frame.groundColor.rgb, frame.skyColor.rgb, n.y * 0.5 + 0.5);
  let r = reflect(-vdir, n);
  let env = mix(frame.groundColor.rgb, frame.skyColor.rgb, smoothstep(-0.4, 0.4, r.y));
  let atten = 1.0 / (dist2 + frame.lightSize * frame.lightSize);
  var col = brdf_ambient(m, n, vdir, hemi, env) + frame.lightColor.rgb * pattern * vis * atten * brdf_direct(m, n, vdir, l);

  switch frame.debugView {
    case 1u: { col = vec3f(vis); }
    case 2u: { col = pattern * smoothstep(0.0, 0.1, ndl); }
    case 3u: {
      // 32×32-texel blocks of the shadow map: where they look big, shadows look blocky.
      let ls = light_space(v.world_pos);
      let c = vec2i(floor(ls.uv * frame.mapSize / 32.0));
      let check = f32((c.x + c.y) & 1);
      let inside = all(ls.uv >= vec2f(0.0)) && all(ls.uv <= vec2f(1.0)) && ls.z > 0.0;
      col = select(vec3f(0.08), mix(vec3f(0.25, 0.3, 0.4), vec3f(0.9, 0.75, 0.5), check), inside) * (0.5 + 0.5 * saturate(ndl));
    }
    case 4u: {
      let ls = light_space(v.world_pos);
      col = vec3f(saturate((ls.z - frame.shadowNear) / 14.0));
    }
    default: {}
  }
  return vec4f(col, 1.0);
}

// --- Light frustum wireframe -----------------------------------------------

struct LineOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
};

@vertex
fn vs_line(@location(0) p: vec4f) -> LineOut {
  var out: LineOut;
  out.position = frame.viewProj * vec4f(p.xyz, 1.0);
  // w carries a brightness: apex edges bright, the far rim dimmer.
  out.color = vec3f(1.0, 0.8, 0.35) * p.w;
  return out;
}

@fragment
fn fs_line(v: LineOut) -> @location(0) vec4f {
  return vec4f(v.color, 1.0);
}

// --- Debug overlays -----------------------------------------------------------

@fragment
fn fs_overlay_map(v: FullscreenOut) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(shadow_map));
  let d = textureLoad(shadow_map, vec2i(v.uv * dims), 0);
  // Linearised, so the perspective map's depth isn't all crammed near 1.
  return vec4f(vec3f(1.0 - saturate((linear_depth(d) - frame.shadowNear) / 14.0)), 1.0);
}

@fragment
fn fs_overlay_cookie(v: FullscreenOut) -> @location(0) vec4f {
  return vec4f(textureSampleLevel(cookie_tex, cookie_smp, v.uv, frame.cookieBlur).rgb, 1.0);
}
