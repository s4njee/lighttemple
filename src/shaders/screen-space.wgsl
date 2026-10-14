// ---------------------------------------------------------------------------
// screen-space.wgsl — view-space geometry from a depth buffer, shared by the
// screen-space passes (src/post/gbuffer.ts, gtao.ts, temporal-accumulate.ts,
// bilateral.ts) and the experiments that use them.
//
// Conventions:
//   - View space is right-handed and the camera looks down -z. `z` is the
//     *positive* linear view depth; 0 means "no geometry here" (background).
//   - uv is [0, 1] with the origin top-left (texture coordinates).
//   - proj_scale = (P[0][0], P[1][1]) of a symmetric perspective projection.
//   - jitter = the TAA sub-pixel offset in NDC that the depth buffer was
//     rendered with ((0, 0) when not jittering).
//
// The packed "geometry" texture (GBuffer.resolve) is rgba32float: view-space
// normal in xyz, z in w, at the resolution of the pass that reads it.
// ---------------------------------------------------------------------------

/// Linear view depth from a [0, 1] perspective depth (WebGPU convention).
fn ss_linear_depth(d: f32, near: f32, far: f32) -> f32 {
  return near * far / (far - d * (far - near));
}

/// View-space position of the surface seen at `uv`, `z` units in front of the camera.
fn ss_view_pos(uv: vec2f, z: f32, proj_scale: vec2f, jitter: vec2f) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) - jitter;
  return vec3f(ndc * z / proj_scale, -z);
}

/// Inverse of ss_view_pos: where a view-space point lands on screen.
fn ss_view_uv(p: vec3f, proj_scale: vec2f, jitter: vec2f) -> vec2f {
  let ndc = p.xy * proj_scale / -p.z + jitter;
  return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

/// Edge-stopping weight between two geometry samples (normal xyz, depth w): 1 for the same
/// surface, falling to 0 across depth discontinuities and creases. Background (w = 0) never matches.
fn ss_geometry_weight(a: vec4f, b: vec4f, depth_sigma: f32, normal_power: f32) -> f32 {
  if (b.w <= 0.0) { return 0.0; }
  let dz = abs(a.w - b.w) / (a.w * depth_sigma);
  let nw = pow(saturate(dot(a.xyz, b.xyz)), normal_power);
  return exp(-dz * dz) * nw;
}
