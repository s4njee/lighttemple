// ---------------------------------------------------------------------------
// temporal-accumulate.wgsl — per-pixel running average of a noisy screen-space
// signal (AO, indirect light) across frames, with reprojection.
//
//  1. Rebuild the pixel's view position from the packed geometry texture and
//     project it with last frame's camera (view -> previous clip).
//  2. Read the history from the four texels around that point. Each texel
//     counts only if last frame's geometry there has the depth this surface
//     should have had ("bilateral" reprojection): a texel that belonged to
//     something else (disocclusion, an edge) is dropped instead of smeared in.
//  3. Blend with weight 1 / n, n = frames of valid history (capped), so a fresh
//     pixel converges as fast as a plain average and a settled one moves slowly.
//     Optionally clip the history to the current 3×3 neighbourhood's spread
//     (for signals whose *value* changes, like moving light).
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, screen-space.wgsl and
// the `Accumulate` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> acc: Accumulate;
@group(0) @binding(1) var cur_signal: texture_2d<f32>;
@group(0) @binding(2) var cur_geometry: texture_2d<f32>;
@group(0) @binding(3) var prev_geometry: texture_2d<f32>;
@group(0) @binding(4) var hist_signal: texture_2d<f32>;
@group(0) @binding(5) var hist_count: texture_2d<f32>;

struct AccumulateOut {
  @location(0) signal: vec4f,
  @location(1) count: vec4f,
};

fn fresh(c: vec4f) -> AccumulateOut {
  return AccumulateOut(c, vec4f(1.0, 0.0, 0.0, 0.0));
}

@fragment
fn fs_accumulate(v: FullscreenOut) -> AccumulateOut {
  let dims = vec2i(textureDimensions(cur_signal));
  let px = vec2i(v.position.xy);
  let c = textureLoad(cur_signal, px, 0);
  let g = textureLoad(cur_geometry, px, 0);
  if (acc.historyValid == 0u || g.w <= 0.0) { return fresh(c); }

  // 1. Where was this surface last frame, and how far from the camera?
  let pos = ss_view_pos((vec2f(px) + 0.5) / vec2f(dims), g.w, acc.projScale, acc.jitter);
  let clip = acc.viewToPrevClip * vec4f(pos, 1.0);
  if (clip.w <= 0.0) { return fresh(c); }
  let prev_uv = vec2f(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
  let expected = clip.w;

  // 2. Bilinear footprint, keeping only texels of the same surface.
  let f = prev_uv * vec2f(dims) - 0.5;
  let base = vec2i(floor(f));
  let fr = f - floor(f);
  var hist = vec4f(0.0);
  var count = 0.0;
  var wsum = 0.0;
  for (var i = 0; i < 4; i++) {
    let o = vec2i(i & 1, i >> 1);
    let q = base + o;
    if (any(q < vec2i(0)) || any(q >= dims)) { continue; }
    let pg = textureLoad(prev_geometry, q, 0);
    if (pg.w <= 0.0 || abs(pg.w - expected) > acc.depthTolerance * expected) { continue; }
    let w = mix(1.0 - fr.x, fr.x, f32(o.x)) * mix(1.0 - fr.y, fr.y, f32(o.y)) + 1e-4;
    hist += w * textureLoad(hist_signal, q, 0);
    count += w * textureLoad(hist_count, q, 0).r;
    wsum += w;
  }
  if (wsum <= 0.0) { return fresh(c); }
  hist /= wsum;
  count /= wsum;

  // 3. Optional clip toward the current neighbourhood (mean ± k·sigma).
  if (acc.clampSigma > 0.0) {
    var m1 = vec4f(0.0);
    var m2 = vec4f(0.0);
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let s = textureLoad(cur_signal, clamp(px + vec2i(x, y), vec2i(0), dims - 1), 0);
        m1 += s;
        m2 += s * s;
      }
    }
    let mean = m1 / 9.0;
    let sigma = sqrt(abs(m2 / 9.0 - mean * mean));
    hist = clamp(hist, mean - acc.clampSigma * sigma, mean + acc.clampSigma * sigma);
  }

  let n = min(count + 1.0, acc.maxFrames);
  return AccumulateOut(mix(hist, c, 1.0 / n), vec4f(n, 0.0, 0.0, 0.0));
}
