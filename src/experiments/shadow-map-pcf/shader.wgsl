// ---------------------------------------------------------------------------
// Directional-light shadow mapping with switchable filters.
// Outputs premultiplied linear HDR (alpha = coverage) into the post chain.
// Requires (prepended): common.wgsl, fullscreen.wgsl, VertexIn, the `Frame`
// struct, shadow.wgsl and brdf.wgsl.
// ---------------------------------------------------------------------------

struct Object {
  model: mat4x4f,
  // rgb = albedo (linear), a = perceptual roughness
  color: vec4f,
  // x = metallic
  material: vec4f,
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
  // Rotating the noise per frame lets TAA integrate it into a smooth result.
  let angle = ign(frag + frame.noiseOffset) * TAU;
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

  let m = Material(obj.color.rgb, obj.color.a, obj.material.x);

  // Hemisphere "environment": sky from above, bounce from the ground below.
  // Diffuse uses the normal; specular uses the reflection vector, softened
  // with roughness, through the split-sum approximation.
  let irradiance = mix(frame.groundColor.rgb, frame.skyColor.rgb, n.y * 0.5 + 0.5);
  let r = reflect(-vdir, n);
  let edge = mix(0.05, 0.8, m.roughness);
  let env = mix(frame.groundColor.rgb, frame.skyColor.rgb, smoothstep(-edge, edge, r.y));
  var col = brdf_ambient(m, n, vdir, irradiance, env) + frame.lightColor.rgb * vis * brdf_direct(m, n, vdir, l);

  if (frame.debugView == 1u) { col = vec3f(vis); }
  if (frame.debugView == 2u) {
    // Light-space depth & out-of-frustum regions.
    col = select(vec3f(0.6, 0.1, 0.1), vec3f(lp.z), shadow_in_bounds(lp));
  }

  // Coverage alpha: the floor fades out radially, and the post chain composites
  // it over the page background *after* tonemapping, so the two match exactly
  // and the floor melts away with no seam. Props in the middle stay opaque.
  let fade = select(0.0, smoothstep(frame.fadeStart, frame.fadeEnd, length(v.world_pos.xz)), frame.debugView == 0u);
  let coverage = 1.0 - fade;
  return vec4f(col * coverage, coverage);
}

// --- Debug overlay: the raw shadow map in a corner ------------------------

@fragment
fn fs_overlay(v: FullscreenOut) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(shadow_map));
  let d = textureLoad(shadow_map, vec2i(v.uv * dims), 0);
  // Stretch contrast: most of the scene lives in a narrow depth band.
  return vec4f(vec3f(pow(saturate(d), 6.0)), 1.0);
}
