// ---------------------------------------------------------------------------
// bilateral.wgsl — edge-aware filters for screen-space signals.
//
//   fs_denoise    (2r+1)² Gaussian at the signal's resolution, each tap
//                 weighted by how well its depth and normal match the centre
//                 (ss_geometry_weight), so noise is averaged within a surface
//                 but never across a silhouette or a crease.
//   fs_upsample   low-res signal -> full resolution: the four nearest low-res
//                 texels, bilinear weights times the same geometry weight
//                 against the full-res depth and normal. If no texel matches
//                 (a thin feature the low-res buffer missed), the one with the
//                 closest depth wins instead of a blurry average.
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, screen-space.wgsl and
// the `Bilateral` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> bl: Bilateral;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var src_geometry: texture_2d<f32>;

@fragment
fn fs_denoise(v: FullscreenOut) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(src));
  let px = vec2i(v.position.xy);
  let c = textureLoad(src, px, 0);
  let g = textureLoad(src_geometry, px, 0);
  if (g.w <= 0.0) { return c; }
  let r = i32(bl.radius);
  let k = -0.5 / (bl.spatialSigma * bl.spatialSigma);
  var sum = c;
  var wsum = 1.0;
  for (var y = -r; y <= r; y++) {
    for (var x = -r; x <= r; x++) {
      let q = px + vec2i(x, y);
      if ((x == 0 && y == 0) || any(q < vec2i(0)) || any(q >= dims)) { continue; }
      let w = exp(f32(x * x + y * y) * k) * ss_geometry_weight(g, textureLoad(src_geometry, q, 0), bl.depthSigma, bl.normalPower);
      sum += w * textureLoad(src, q, 0);
      wsum += w;
    }
  }
  return sum / wsum;
}

// --- Upsample ---------------------------------------------------------------

@group(0) @binding(3) var full_depth: texture_depth_2d;
@group(0) @binding(4) var full_normal: texture_2d<f32>;

@fragment
fn fs_upsample(v: FullscreenOut) -> @location(0) vec4f {
  let px = vec2i(v.position.xy);
  let d = textureLoad(full_depth, px, 0);
  if (d >= 1.0) { return bl.background; }
  let g = vec4f(normalize(textureLoad(full_normal, px, 0).xyz), ss_linear_depth(d, bl.near, bl.far));

  let low_dims = vec2i(textureDimensions(src));
  let full_dims = vec2f(textureDimensions(full_depth));
  let f = (vec2f(px) + 0.5) * vec2f(low_dims) / full_dims - 0.5;
  let base = vec2i(floor(f));
  let fr = f - floor(f);
  var sum = vec4f(0.0);
  var wsum = 0.0;
  var nearest = bl.background;
  var nearest_dz = 1e30;
  for (var i = 0; i < 4; i++) {
    let o = vec2i(i & 1, i >> 1);
    let q = clamp(base + o, vec2i(0), low_dims - 1);
    let gq = textureLoad(src_geometry, q, 0);
    let s = textureLoad(src, q, 0);
    let w = mix(1.0 - fr.x, fr.x, f32(o.x)) * mix(1.0 - fr.y, fr.y, f32(o.y)) * ss_geometry_weight(g, gq, bl.depthSigma, bl.normalPower);
    sum += w * s;
    wsum += w;
    let dz = select(1e30, abs(gq.w - g.w), gq.w > 0.0);
    if (dz < nearest_dz) {
      nearest_dz = dz;
      nearest = s;
    }
  }
  return select(nearest, sum / max(wsum, 1e-8), wsum > 1e-3);
}
