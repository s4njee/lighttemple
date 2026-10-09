// ---------------------------------------------------------------------------
// common.wgsl — constants, colour pipeline and noise shared by experiments.
// ---------------------------------------------------------------------------

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn saturate3(x: vec3f) -> vec3f { return clamp(x, vec3f(0.0), vec3f(1.0)); }

// --- Colour ----------------------------------------------------------------
// Canvas formats (bgra8unorm / rgba8unorm) are not sRGB, so we encode by hand.

fn linear_to_srgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn srgb_to_linear(c: vec3f) -> vec3f {
  let lo = c / 12.92;
  let hi = pow((c + 0.055) / 1.055, vec3f(2.4));
  return select(hi, lo, c <= vec3f(0.04045));
}

fn luminance(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// Narkowicz 2015 ACES fit. Punchy, slightly saturated highlights.
fn tonemap_aces(x: vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return saturate3((x * (a * x + b)) / (x * (c * x + d) + e));
}

// AgX-style approximation (Benjamin Wrensch's fit). Softer, desaturates gracefully
// towards white — tends to look more "photographic" for bright lights.
fn agx_contrast(x: vec3f) -> vec3f {
  let x2 = x * x; let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn tonemap_agx(c: vec3f) -> vec3f {
  let m_in = mat3x3f(
    vec3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
    vec3f(0.0784335999999992, 0.878468636469772, 0.0784336),
    vec3f(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
  let m_out = mat3x3f(
    vec3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
    vec3f(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
    vec3f(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
  let min_ev = -12.47393; let max_ev = 4.026069;
  var v = m_in * max(c, vec3f(1e-10));
  v = clamp(log2(v), vec3f(min_ev), vec3f(max_ev));
  v = (v - min_ev) / (max_ev - min_ev);
  v = agx_contrast(v);
  v = m_out * v;
  // agx_contrast outputs display-encoded values; return linear so callers can encode uniformly.
  return srgb_to_linear(saturate3(v));
}

// mode: 0 = none (clamp), 1 = ACES, 2 = AgX
fn tonemap(c: vec3f, mode: u32) -> vec3f {
  switch mode {
    case 1u: { return tonemap_aces(c); }
    case 2u: { return tonemap_agx(c); }
    default: { return saturate3(c); }
  }
}

// --- Noise -----------------------------------------------------------------

// Jimenez 2014 interleaved gradient noise: cheap, well-distributed per-pixel jitter.
fn ign(p: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// Final display encode: tonemapped linear -> sRGB, plus +-0.5 LSB triangular dither
// so smooth light falloffs don't band on 8-bit swapchains.
fn display_encode(linear_rgb: vec3f, frag: vec2f) -> vec3f {
  let srgb = linear_to_srgb(saturate3(linear_rgb));
  let n = hash12(frag) + hash12(frag + vec2f(17.13, 3.71)) - 1.0;
  return srgb + n / 255.0;
}
