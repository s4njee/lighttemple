// ---------------------------------------------------------------------------
// fullscreen.wgsl — a single oversized triangle covering the viewport.
// Draw with `pass.draw(3)`; no vertex buffers needed.
// ---------------------------------------------------------------------------

struct FullscreenOut {
  @builtin(position) position: vec4f,
  // uv in [0,1], origin top-left (matches texture coordinates).
  @location(0) uv: vec2f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> FullscreenOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var out: FullscreenOut;
  out.position = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(p.x, 1.0 - p.y);
  return out;
}
