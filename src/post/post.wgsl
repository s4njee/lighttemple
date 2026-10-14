// ---------------------------------------------------------------------------
// post.wgsl — bloom chain + final composite.
// Requires (prepended): common.wgsl, fullscreen.wgsl, the `Post` struct.
//
// Scene input is premultiplied linear HDR (rgba16float); alpha is coverage,
// letting experiments composite over a display-space background colour
// (fs_composite) or write premultiplied alpha for a transparent canvas
// (fs_composite_premultiplied).
// ---------------------------------------------------------------------------

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var<uniform> post: Post;

fn tap(uv: vec2f, offset: vec2f, texel: vec2f) -> vec3f {
  return textureSampleLevel(src_tex, lin, uv + offset * texel, 0.0).rgb;
}

// --- Downsample: 13-tap "Call of Duty: Advanced Warfare" filter (Jimenez 2014) ---

struct Taps13 { a: vec3f, b: vec3f, c: vec3f, d: vec3f, e: vec3f, f: vec3f, g: vec3f, h: vec3f, i: vec3f, j: vec3f, k: vec3f, l: vec3f, m: vec3f };

fn taps13(uv: vec2f) -> Taps13 {
  let t = 1.0 / vec2f(textureDimensions(src_tex));
  return Taps13(
    tap(uv, vec2f(-2.0, 2.0), t), tap(uv, vec2f(0.0, 2.0), t), tap(uv, vec2f(2.0, 2.0), t),
    tap(uv, vec2f(-2.0, 0.0), t), tap(uv, vec2f(0.0, 0.0), t), tap(uv, vec2f(2.0, 0.0), t),
    tap(uv, vec2f(-2.0, -2.0), t), tap(uv, vec2f(0.0, -2.0), t), tap(uv, vec2f(2.0, -2.0), t),
    tap(uv, vec2f(-1.0, 1.0), t), tap(uv, vec2f(1.0, 1.0), t), tap(uv, vec2f(-1.0, -1.0), t), tap(uv, vec2f(1.0, -1.0), t));
}

@fragment
fn fs_down(v: FullscreenOut) -> @location(0) vec4f {
  let s = taps13(v.uv);
  let c = s.e * 0.125 + (s.a + s.c + s.g + s.i) * 0.03125 + (s.b + s.d + s.f + s.h) * 0.0625 + (s.j + s.k + s.l + s.m) * 0.125;
  return vec4f(c, 1.0);
}

fn karis(c: vec3f) -> f32 { return 1.0 / (1.0 + luminance(c)); }

// First downsample only: weight each 2x2 group by 1/(1+luma) (Karis average)
// so single very bright pixels can't produce flickering "firefly" blooms.
@fragment
fn fs_down_karis(v: FullscreenOut) -> @location(0) vec4f {
  let s = taps13(v.uv);
  let g0 = (s.j + s.k + s.l + s.m) * 0.25;
  let g1 = (s.a + s.b + s.d + s.e) * 0.25;
  let g2 = (s.b + s.c + s.e + s.f) * 0.25;
  let g3 = (s.d + s.e + s.g + s.h) * 0.25;
  let g4 = (s.e + s.f + s.h + s.i) * 0.25;
  let w0 = 0.5 * karis(g0);
  let w1 = 0.125 * karis(g1);
  let w2 = 0.125 * karis(g2);
  let w3 = 0.125 * karis(g3);
  let w4 = 0.125 * karis(g4);
  let c = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  return vec4f(max(c, vec3f(0.0)), 1.0);
}

// --- Upsample: 3x3 tent, additively blended into the next larger level -------

@fragment
fn fs_up(v: FullscreenOut) -> @location(0) vec4f {
  let t = post.bloomRadius / vec2f(textureDimensions(src_tex));
  let uv = v.uv;
  var c = tap(uv, vec2f(0.0, 0.0), t) * 4.0;
  c += (tap(uv, vec2f(0.0, 1.0), t) + tap(uv, vec2f(-1.0, 0.0), t) + tap(uv, vec2f(1.0, 0.0), t) + tap(uv, vec2f(0.0, -1.0), t)) * 2.0;
  c += tap(uv, vec2f(-1.0, 1.0), t) + tap(uv, vec2f(1.0, 1.0), t) + tap(uv, vec2f(-1.0, -1.0), t) + tap(uv, vec2f(1.0, -1.0), t);
  return vec4f(c / 16.0, 1.0);
}

// --- Composite ---------------------------------------------------------------

/// Bloom, exposure, tonemapping, vignette and grain on an *unpremultiplied* scene colour.
fn grade(scene: vec3f, uv: vec2f, frag: vec2f) -> vec3f {
  var c = scene;
  if (post.bloomStrength > 0.0) {
    // Each level was summed once on the way up; normalise by the level count.
    let bloom = textureSampleLevel(bloom_tex, lin, uv, 0.0).rgb / post.bloomLevels;
    c = mix(c, bloom, post.bloomStrength);
  }
  c = tonemap(c * post.exposure, post.tonemap);
  // Vignette in display-linear space: a gentle, round falloff.
  let q = (uv - 0.5) * vec2f(post.aspect, 1.0);
  c *= mix(1.0, smoothstep(1.2, 0.2, length(q)), post.vignette);
  // Film grain scaled by luminance so blacks stay clean.
  let g = hash12(frag + fract(post.time) * 1000.0) - 0.5;
  c += g * post.grain * sqrt(luminance(c) + 1e-4);
  return c;
}

@fragment
fn fs_composite(v: FullscreenOut) -> @location(0) vec4f {
  let frag = v.position.xy;
  let hdr = textureLoad(src_tex, vec2i(frag), 0);
  let a = saturate(hdr.a);
  var c = select(vec3f(0.0), hdr.rgb / max(hdr.a, 1e-5), hdr.a > 1e-5);
  if (post.raw == 0u) { c = grade(c, v.uv, frag); }

  let out = mix(post.background, c, a);
  // Flat background stays undithered so its colour is exact (e.g. #FF0000).
  let encoded = mix(linear_to_srgb(saturate3(out)), display_encode(out, frag), a);
  return vec4f(encoded, 1.0);
}

// Premultiplied output for a transparent canvas (alphaMode 'premultiplied') over live HTML.
// The browser composites `rgb + (1 - a) * page` in sRGB space, so:
//   - colour is graded and dithered *unpremultiplied*, clamped to [0, 1], then multiplied by
//     alpha: rgb <= alpha always holds, and fully transparent pixels stay exactly (0, 0, 0, 0)
//     (bloom, grain and dither never leak outside coverage or speckle the alpha);
//   - alpha itself is never dithered. An 8-bit alpha step changes the page by at most one LSB;
//   - alphaMode 1 ("linear") treats dark pixels as transmittance: black with alpha a is written
//     with alpha 1 - (1 - a)^(1/2.2), which darkens *any* page colour by what the opaque path's
//     linear-light `mix(background, 0, a)` would (to the gamma-2.2 approximation), so a shadow
//     looks the same over HTML as over a flat page colour. Bright pixels (anti-aliased edges of
//     lit objects) keep their coverage alpha, where the plain sRGB blend is the better match;
//     the two are mixed by the pixel's display luminance.
//     alphaMode 0 ("coverage") writes scene alpha unchanged, the plain CSS behaviour.
@fragment
fn fs_composite_premultiplied(v: FullscreenOut) -> @location(0) vec4f {
  let frag = v.position.xy;
  let hdr = textureLoad(src_tex, vec2i(frag), 0);
  let a = saturate(hdr.a);
  if (a <= 0.0) { return vec4f(0.0); }
  var c = hdr.rgb / max(hdr.a, 1e-5);
  if (post.raw == 0u) { c = grade(c, v.uv, frag); }
  let encoded = saturate3(display_encode(c, frag));
  // The weight uses the undithered colour, so the alpha stays free of dither noise.
  let transmittance_alpha = mix(1.0 - pow(1.0 - a, 1.0 / 2.2), a, luminance(linear_to_srgb(saturate3(c))));
  let alpha = select(a, transmittance_alpha, post.alphaMode == 1u);
  return vec4f(encoded * alpha, alpha);
}
