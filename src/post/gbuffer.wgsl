// ---------------------------------------------------------------------------
// gbuffer.wgsl — packs the G-buffer's depth + view normal into one geometry
// texture (view normal xyz, linear depth w) at full or half resolution.
//
// Half resolution keeps one real sample per 2×2 block rather than averaging
// (an average depth is a surface that doesn't exist). Blocks alternate in a
// checkerboard between the nearest and the farthest of their four samples,
// so both sides of a depth edge survive the downsample and a bilateral
// upsample always finds a matching neighbour.
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, screen-space.wgsl and
// the `GBufferResolve` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> resolve: GBufferResolve;
@group(0) @binding(1) var gb_depth: texture_depth_2d;
@group(0) @binding(2) var gb_normal: texture_2d<f32>;

@fragment
fn fs_resolve(v: FullscreenOut) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(gb_depth));
  let px = vec2i(v.position.xy);
  var best = px;
  var best_d = textureLoad(gb_depth, px, 0);
  if (resolve.factor > 1u) {
    let base = px * 2;
    let farthest = ((px.x + px.y) & 1) == 1;
    best = min(base, dims - 1);
    best_d = textureLoad(gb_depth, best, 0);
    for (var i = 1; i < 4; i++) {
      let q = min(base + vec2i(i & 1, i >> 1), dims - 1);
      let d = textureLoad(gb_depth, q, 0);
      if ((farthest && d > best_d) || (!farthest && d < best_d)) {
        best = q;
        best_d = d;
      }
    }
  }
  if (best_d >= 1.0) { return vec4f(0.0, 0.0, 1.0, 0.0); }
  let n = textureLoad(gb_normal, best, 0).xyz;
  return vec4f(normalize(n), ss_linear_depth(best_d, resolve.near, resolve.far));
}
