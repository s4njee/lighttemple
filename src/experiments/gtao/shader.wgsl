// ---------------------------------------------------------------------------
// Ground-Truth AO
// Outputs linear HDR (alpha = coverage) into the shared post chain.
// Requires (prepended): common.wgsl, fullscreen.wgsl, the `Params` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> u: Params;

@fragment
fn fs_main(v: FullscreenOut) -> @location(0) vec4f {
  // World space: y in [-1, 1] (up), x in [-aspect, aspect].
  let p = (v.position.xy - 0.5 * u.resolution) / u.resolution.y * vec2f(2.0, -2.0);

  // A point light hovering over a flat floor: inverse-square falloff x Lambert.
  let to_light = vec3f(u.light.xy - p, u.light.z);
  let dist2 = dot(to_light, to_light);
  let ndl = u.light.z * inverseSqrt(dist2);
  let albedo = vec3f(0.6, 0.57, 0.53);
  let col = albedo / PI * u.lightColor.rgb * u.intensity * ndl / dist2 + albedo * 0.02;
  return vec4f(col, 1.0);
}
