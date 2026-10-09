// ---------------------------------------------------------------------------
// shadow.wgsl — shadow-map filtering for an orthographic (directional) light.
//
// Bind a depth texture + comparison sampler and call one of the filters with a
// light-space position `lp` = (uv.xy, depth). All taps use the *Level variant
// so they are legal in non-uniform control flow.
//
// Requires these bindings to be declared by the including shader:
//   var shadow_map: texture_depth_2d;
//   var shadow_cmp: sampler_comparison;
// ---------------------------------------------------------------------------

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

fn shadow_in_bounds(lp: vec3f) -> bool {
  return all(lp.xy >= vec2f(0.0)) && all(lp.xy <= vec2f(1.0)) && lp.z <= 1.0;
}

/// Single hardware comparison (with linear comparison sampler this is 2x2 bilinear PCF).
fn shadow_hard(lp: vec3f) -> f32 {
  return textureSampleCompareLevel(shadow_map, shadow_cmp, lp.xy, lp.z);
}

/// (2r+1)^2 grid of bilinear comparison taps, spaced one texel apart.
fn shadow_pcf_grid(lp: vec3f, texel: f32, r: i32) -> f32 {
  var sum = 0.0;
  for (var y = -r; y <= r; y++) {
    for (var x = -r; x <= r; x++) {
      sum += textureSampleCompareLevel(shadow_map, shadow_cmp, lp.xy + vec2f(f32(x), f32(y)) * texel, lp.z);
    }
  }
  let n = f32(2 * r + 1);
  return sum / (n * n);
}

/// 16 Poisson taps in a disc of `radius_uv`, rotated per pixel by `angle`.
/// Trades banding for fine noise (which grain/TAA hide well).
fn shadow_poisson(lp: vec3f, radius_uv: f32, angle: f32) -> f32 {
  let c = cos(angle); let s = sin(angle);
  let rot = mat2x2f(c, s, -s, c);
  var disk = POISSON_16;
  var sum = 0.0;
  for (var i = 0; i < 16; i++) {
    sum += textureSampleCompareLevel(shadow_map, shadow_cmp, lp.xy + rot * disk[i] * radius_uv, lp.z);
  }
  return sum / 16.0;
}

/// Percentage-closer soft shadows (Fernando 2005) for an orthographic light.
///   light_tan:    tangent of the light's angular radius (sun ≈ 0.0046)
///   depth_range:  world-space distance covered by depth 0..1
///   world_to_uv:  1 / (world width of the shadow frustum)
fn shadow_pcss(lp: vec3f, texel: f32, light_tan: f32, depth_range: f32, world_to_uv: f32, angle: f32) -> f32 {
  let c = cos(angle); let s = sin(angle);
  let rot = mat2x2f(c, s, -s, c);
  var disk = POISSON_16;
  let dims = vec2f(textureDimensions(shadow_map));

  // 1. Blocker search: the region of the map that could hold occluders for this receiver.
  let search_uv = clamp(light_tan * lp.z * depth_range * world_to_uv, 2.0 * texel, 48.0 * texel);
  var blocker_sum = 0.0;
  var blockers = 0.0;
  for (var i = 0; i < 16; i++) {
    let uv = lp.xy + rot * disk[i] * search_uv;
    let px = clamp(vec2i(uv * dims), vec2i(0), vec2i(dims) - 1);
    let d = textureLoad(shadow_map, px, 0);
    if (d < lp.z) {
      blocker_sum += d;
      blockers += 1.0;
    }
  }
  if (blockers == 0.0) { return 1.0; }
  let avg_blocker = blocker_sum / blockers;

  // 2. Penumbra width from similar triangles; for a directional light it is
  //    proportional to the receiver–blocker distance only.
  let penumbra_uv = light_tan * (lp.z - avg_blocker) * depth_range * world_to_uv;
  let radius = clamp(penumbra_uv, texel, 64.0 * texel);

  // 3. Filter with that radius.
  return shadow_poisson(lp, radius, angle);
}
