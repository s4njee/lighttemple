// ---------------------------------------------------------------------------
// contact-shadow.wgsl — screen-space contact shadows.
//
// From a shaded point, march a short distance toward the light, project each
// step into the screen and test it against the scene's depth buffer. A step
// that lands a little *behind* the stored surface is occluded. This recovers
// the thin, dark contact line that shadow maps lose to their texel size and
// bias, but it only knows what is on screen and sees every surface as a shell
// `thickness` deep.
//
// Requires this binding to be declared by the including shader: a single-
// sampled depth buffer of the *same* (jittered) view-projection:
//   var scene_depth: texture_depth_2d;
// ---------------------------------------------------------------------------

struct ContactShadowParams {
  /// View-projection the depth buffer was rendered with (perspective).
  view_proj: mat4x4f,
  /// Depth buffer size in pixels.
  screen: vec2f,
  /// Camera near / far planes, to linearise the depth buffer.
  near: f32,
  far: f32,
  /// March distance in world units.
  length: f32,
  /// How deep each surface is assumed to be, in world units.
  thickness: f32,
  steps: i32,
  /// 0..1 per-pixel offset of the first step (noise; rotate per frame under TAA).
  jitter: f32,
};

/// Linear view distance from a [0, 1] perspective depth (WebGPU convention).
fn contact_linear_depth(d: f32, near: f32, far: f32) -> f32 {
  return near * far / (far - d * (far - near));
}

/// 1 = lit, 0 = in contact shadow. `to_light` is a unit vector toward the light.
fn contact_shadow(world_pos: vec3f, to_light: vec3f, p: ContactShadowParams) -> f32 {
  // Steps that land in the shaded pixel itself are skipped: at silhouettes the depth stored there can
  // belong to the surface in front, which would shadow the whole edge.
  let origin = p.view_proj * vec4f(world_pos, 1.0);
  let own_px = vec2i((vec2f(origin.x, -origin.y) / origin.w * 0.5 + 0.5) * p.screen);
  let n = max(p.steps, 1);
  for (var i = 0; i < n; i++) {
    let t = (f32(i) + p.jitter) / f32(n) * p.length;
    let clip = p.view_proj * vec4f(world_pos + to_light * t, 1.0);
    // For a perspective projection clip.w is the point's linear view depth.
    if (clip.w <= p.near) { break; }
    let uv = vec2f(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) { break; }
    let px = vec2i(uv * p.screen);
    if (all(px == own_px)) { continue; }
    let scene = contact_linear_depth(textureLoad(scene_depth, px, 0), p.near, p.far);
    let behind = clip.w - scene;
    // A small depth-proportional bias keeps the ray from hitting its own surface.
    if (behind > 0.002 * clip.w && behind < p.thickness) {
      // Fade hits near the end of the ray, so its reach doesn't show as a hard edge.
      let f = t / p.length;
      return f * f;
    }
  }
  return 1.0;
}
