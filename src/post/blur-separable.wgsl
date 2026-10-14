// ---------------------------------------------------------------------------
// One direction of a separable Gaussian blur. Exact texel loads (no sampler),
// so it works on any float format, including unfilterable rgba32float.
// Requires (prepended): fullscreen.wgsl and the `Blur` uniform struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> blur: Blur;
@group(0) @binding(1) var src: texture_2d<f32>;

@fragment
fn fs_blur(v: FullscreenOut) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(src));
  let px = vec2i(v.position.xy);
  let dir = vec2i(blur.direction);
  let r = i32(blur.radius);
  let k = -0.5 / (blur.sigma * blur.sigma);
  var sum = vec4f(0.0);
  var weight = 0.0;
  for (var i = -r; i <= r; i++) {
    let w = exp(f32(i * i) * k);
    sum += w * textureLoad(src, clamp(px + dir * i, vec2i(0), dims - 1), 0);
    weight += w;
  }
  return sum / weight;
}
