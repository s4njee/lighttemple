// ---------------------------------------------------------------------------
// Directional-light shadow mapping with switchable filters.
// Requires (prepended): common.wgsl, VertexIn, the `Frame` struct, shadow.wgsl
// is appended after the bindings below.
// ---------------------------------------------------------------------------

struct Object {
  model: mat4x4f,
  // rgb = albedo (linear), a = gloss 0..1
  color: vec4f,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(3) var shadow_cmp: sampler_comparison;

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
  // Objects use uniform scale, so the upper 3x3 is fine for normals.
  out.normal = normalize((m * vec4f(v.normal, 0.0)).xyz);
  out.id = id;
  return out;
}

fn light_space(world_pos: vec3f, n: vec3f, ndl: f32) -> vec3f {
  // Normal-offset bias: push the lookup along the normal, more at grazing angles.
  let offset = n * frame.normalOffset * frame.shadowTexelWorld * (1.0 - ndl * 0.5);
  let clip = frame.lightViewProj * vec4f(world_pos + offset, 1.0);
  let ndc = clip.xyz / clip.w;
  return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z - frame.depthBias);
}

fn sample_shadow(lp: vec3f, frag: vec2f) -> f32 {
  if (!shadow_in_bounds(lp)) { return 1.0; }
  let texel = 1.0 / f32(textureDimensions(shadow_map).x);
  let angle = ign(frag) * TAU;
  let world_to_uv = 1.0 / (2.0 * frame.shadowExtent);
  switch frame.shadowMode {
    case 0u: { return 1.0; }
    case 1u: { return shadow_hard(lp); }
    case 2u: { return shadow_pcf_grid(lp, texel, i32(frame.pcfRadius)); }
    case 3u: { return shadow_poisson(lp, frame.pcfRadius * texel * 1.5, angle); }
    default: { return shadow_pcss(lp, texel, frame.lightTan, frame.shadowDepthRange, world_to_uv, angle); }
  }
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4f {
  let obj = objects[v.id];
  let n = normalize(v.normal);
  let l = -frame.lightDir;
  let vdir = normalize(frame.cameraPos - v.world_pos);
  let ndl = saturate(dot(n, l));

  let lp = light_space(v.world_pos, n, ndl);
  var vis = sample_shadow(lp, v.position.xy);
  // Terminator fix: fade shadow lookups on surfaces facing away from the light
  // (they are already dark from N·L, and filtering there only adds acne).
  vis *= smoothstep(0.0, 0.08, ndl);

  let albedo = obj.color.rgb;
  let gloss = obj.color.a;
  let shininess = mix(8.0, 256.0, gloss * gloss);
  let h = normalize(l + vdir);
  let spec = pow(saturate(dot(n, h)), shininess) * (shininess + 8.0) / (8.0 * PI) * mix(0.02, 0.25, gloss);

  // Hemisphere ambient: sky from above, bounce from the ground below.
  let hemi = mix(frame.groundColor.rgb, frame.skyColor.rgb, n.y * 0.5 + 0.5);
  var col = albedo * hemi + frame.lightColor.rgb * ndl * vis * (albedo / PI + spec);

  col *= frame.exposure;
  if (frame.debugView == 1u) { col = vec3f(vis); }
  if (frame.debugView == 2u) {
    // Light-space depth & out-of-frustum regions.
    col = select(vec3f(0.6, 0.1, 0.1), vec3f(lp.z), shadow_in_bounds(lp));
  }
  var mapped = select(col, tonemap(col, frame.tonemap), frame.debugView == 0u);

  // Radial fade on the world XZ plane, applied after tonemapping so it matches
  // the clear colour exactly: the floor melts into the background with no seam
  // while the props in the middle stay untouched.
  if (frame.debugView == 0u) {
    let fade = smoothstep(frame.fadeStart, frame.fadeEnd, length(v.world_pos.xz));
    mapped = mix(mapped, frame.fogColor.rgb, fade);
  }
  return vec4f(display_encode(mapped, v.position.xy), 1.0);
}

// --- Debug overlay: the raw shadow map in a corner ------------------------

@fragment
fn fs_overlay(v: FullscreenOut) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(shadow_map));
  let d = textureLoad(shadow_map, vec2i(v.uv * dims), 0);
  // Stretch contrast: most of the scene lives in a narrow depth band.
  let c = pow(saturate(d), 6.0);
  return vec4f(linear_to_srgb(vec3f(c)), 1.0);
}
