// ---------------------------------------------------------------------------
// shadow-moments.wgsl — variance (VSM) and exponential variance (EVSM)
// shadow maps. Pure functions, no bindings: write moments with *_moments() in
// the light pass, blur/filter them like any texture, then call *_visibility()
// with the filtered moments and the receiver's light-space depth.
//
// Depths are in [0, 1] (orthographic light: linear in world space).
// Requires common.wgsl (saturate).
// ---------------------------------------------------------------------------

/// VSM moments (d, d²). The second moment gets the depth's variance across the
/// pixel footprint (Lauritzen 2007), which keeps sloped receivers from acne.
fn vsm_moments(d: f32, ddx: f32, ddy: f32) -> vec2f {
  return vec2f(d, d * d + 0.25 * (ddx * ddx + ddy * ddy));
}

/// Chebyshev's one-tailed upper bound on P(occluder depth >= t): an upper bound
/// on the lit fraction. Exact for one planar occluder over one planar receiver;
/// too bright ("light bleeding") when several occluders overlap.
fn chebyshev_upper_bound(m: vec2f, t: f32, min_variance: f32) -> f32 {
  if (t <= m.x) { return 1.0; }
  let variance = max(m.y - m.x * m.x, min_variance);
  let d = t - m.x;
  return variance / (variance + d * d);
}

/// Light-bleeding reduction: maps [amount, 1] to [0, 1], cutting off the
/// bound's weak tail. Costs over-darkened, harder penumbrae as `amount` grows.
fn reduce_light_bleeding(p: f32, amount: f32) -> f32 {
  return saturate((p - amount) / (1.0 - amount));
}

fn vsm_visibility(m: vec2f, t: f32, min_variance: f32, bleed: f32) -> f32 {
  return reduce_light_bleeding(chebyshev_upper_bound(m, t, min_variance), bleed);
}

/// EVSM depth warp (Lauritzen & McCool 2008): depth remapped to [-1, 1], then
/// exp(+c₊·d) and -exp(-c₋·d). `exps` = (c₊, c₋). Keep e^(2c) inside the
/// storage format: c <= ~42 for 32-bit floats, <= ~5.54 for 16-bit.
fn evsm_warp(d: f32, exps: vec2f) -> vec2f {
  let w = 2.0 * d - 1.0;
  return vec2f(exp(exps.x * w), -exp(-exps.y * w));
}

/// Four EVSM moments: (pos, pos², neg, neg²).
fn evsm_moments(d: f32, exps: vec2f) -> vec4f {
  let w = evsm_warp(d, exps);
  return vec4f(w.x, w.x * w.x, w.y, w.y * w.y);
}

/// EVSM visibility: the tighter of the positive and negative Chebyshev bounds.
/// `min_variance` is in [0, 1] depth units and is scaled into each warped space.
fn evsm_visibility(m: vec4f, t: f32, exps: vec2f, min_variance: f32, bleed: f32) -> f32 {
  let w = evsm_warp(t, exps);
  // d(warp)/d(depth) = 2c·warp, so a depth-space deviation σ becomes 2c·warp·σ.
  let pos_scale = 2.0 * exps.x * w.x;
  let neg_scale = 2.0 * exps.y * w.y;
  let p_pos = chebyshev_upper_bound(m.xy, w.x, min_variance * pos_scale * pos_scale);
  let p_neg = chebyshev_upper_bound(m.zw, w.y, min_variance * neg_scale * neg_scale);
  return reduce_light_bleeding(min(p_pos, p_neg), bleed);
}
