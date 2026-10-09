// ---------------------------------------------------------------------------
// taa.wgsl — temporal anti-aliasing / accumulation resolve.
//
//  1. Reproject: reconstruct each pixel's world position from depth (using the
//     closest depth in a 3x3 block, so edges of near objects win) and project
//     it with last frame's view-projection. Handles camera motion; object
//     motion is left to the neighbourhood clip (no velocity buffer yet).
//  2. Sample history with a 9-tap Catmull-Rom filter (sharper than bilinear,
//     so repeated resampling doesn't blur the image).
//  3. Clip history toward the current 3x3 neighbourhood's colour distribution
//     (variance clipping in YCoCg) to reject stale or disoccluded history.
//  4. Blend in a Karis "luma-weighted" space so bright HDR pixels don't flicker.
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, the `Taa` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var cur_tex: texture_2d<f32>;
@group(0) @binding(1) var hist_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var<uniform> taa: Taa;

fn rgb_to_ycocg(c: vec3f) -> vec3f {
  return vec3f(
    dot(c, vec3f(0.25, 0.5, 0.25)),
    dot(c, vec3f(0.5, 0.0, -0.5)),
    dot(c, vec3f(-0.25, 0.5, -0.25)));
}

fn ycocg_to_rgb(c: vec3f) -> vec3f {
  return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Reversible luma weighting (Karis 2014): compresses HDR before blending.
fn karis_fwd(c: vec3f) -> vec3f { return c / (1.0 + luminance(c)); }
fn karis_inv(c: vec3f) -> vec3f { return c / max(1.0 - luminance(c), 1e-4); }

fn sample_catmull_rom(uv: vec2f, size: vec2f) -> vec4f {
  let sample_pos = uv * size;
  let tex_pos1 = floor(sample_pos - 0.5) + 0.5;
  let f = sample_pos - tex_pos1;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let offset12 = w2 / w12;
  let p0 = (tex_pos1 - 1.0) / size;
  let p3 = (tex_pos1 + 2.0) / size;
  let p12 = (tex_pos1 + offset12) / size;

  var r = vec4f(0.0);
  r += textureSampleLevel(hist_tex, lin, vec2f(p0.x, p0.y), 0.0) * w0.x * w0.y;
  r += textureSampleLevel(hist_tex, lin, vec2f(p12.x, p0.y), 0.0) * w12.x * w0.y;
  r += textureSampleLevel(hist_tex, lin, vec2f(p3.x, p0.y), 0.0) * w3.x * w0.y;
  r += textureSampleLevel(hist_tex, lin, vec2f(p0.x, p12.y), 0.0) * w0.x * w12.y;
  r += textureSampleLevel(hist_tex, lin, vec2f(p12.x, p12.y), 0.0) * w12.x * w12.y;
  r += textureSampleLevel(hist_tex, lin, vec2f(p3.x, p12.y), 0.0) * w3.x * w12.y;
  r += textureSampleLevel(hist_tex, lin, vec2f(p0.x, p3.y), 0.0) * w0.x * w3.y;
  r += textureSampleLevel(hist_tex, lin, vec2f(p12.x, p3.y), 0.0) * w12.x * w3.y;
  r += textureSampleLevel(hist_tex, lin, vec2f(p3.x, p3.y), 0.0) * w3.x * w3.y;
  return max(r, vec4f(0.0));
}

// Clip `h` toward the centre of the AABB [lo, hi] (cheaper and less "boxy"
// than a per-channel clamp; Playdead/INSIDE style).
fn clip_aabb(lo: vec3f, hi: vec3f, h: vec3f) -> vec3f {
  let center = 0.5 * (hi + lo);
  let extents = 0.5 * (hi - lo) + 1e-5;
  let v = h - center;
  let unit = abs(v / extents);
  let m = max(unit.x, max(unit.y, unit.z));
  return select(h, center + v / m, m > 1.0);
}

@fragment
fn fs_resolve(v: FullscreenOut) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(cur_tex));
  let px = vec2i(v.position.xy);
  let cur = textureLoad(cur_tex, px, 0);
  let c = rgb_to_ycocg(karis_fwd(cur.rgb));

  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  var closest = 1.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = clamp(px + vec2i(x, y), vec2i(0), dims - 1);
      let s = rgb_to_ycocg(karis_fwd(textureLoad(cur_tex, q, 0).rgb));
      m1 += s;
      m2 += s * s;
      closest = min(closest, textureLoad(depth_tex, q, 0));
    }
  }
  let mean = m1 / 9.0;
  let sigma = sqrt(abs(m2 / 9.0 - mean * mean));

  if (taa.historyValid == 0u) { return cur; }

  // Reproject with the (unjittered) matrices.
  let uv = v.position.xy / vec2f(dims);
  let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, closest, 1.0);
  let world = taa.invViewProj * ndc;
  let prev_clip = taa.prevViewProj * vec4f(world.xyz / world.w, 1.0);
  let prev_uv = vec2f(prev_clip.x / prev_clip.w * 0.5 + 0.5, 0.5 - prev_clip.y / prev_clip.w * 0.5);
  if (any(prev_uv < vec2f(0.0)) || any(prev_uv > vec2f(1.0))) { return cur; }

  let hist = sample_catmull_rom(prev_uv, vec2f(dims));
  let h = clip_aabb(mean - taa.gamma * sigma, mean + taa.gamma * sigma, rgb_to_ycocg(karis_fwd(hist.rgb)));

  // Trust history less while things move fast (less smearing, slightly more aliasing).
  let motion_px = length((prev_uv - uv) * vec2f(dims));
  let feedback = mix(taa.feedback, taa.feedback * 0.8, saturate(motion_px / 4.0));
  let rgb = karis_inv(ycocg_to_rgb(mix(c, h, feedback)));
  let a = mix(cur.a, hist.a, feedback);
  return vec4f(max(rgb, vec3f(0.0)), a);
}
