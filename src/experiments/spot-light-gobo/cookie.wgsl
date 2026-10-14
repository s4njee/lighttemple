// ---------------------------------------------------------------------------
// Procedural gobos ("cookies"): what the spot light shines through.
// Rendered into a mipmapped rgba8unorm texture; the mip chain doubles as the
// projector's focus (sample a blurrier level for a softer, out-of-focus gobo).
// Requires (prepended): common.wgsl, fullscreen.wgsl, noise.wgsl, sdf.wgsl
// (rot2), the `Cookie` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> ck: Cookie;

/// One layer of leaves: max coverage of pointed ellipses scattered on a jittered grid.
fn leaf_layer(p: vec2f, scale: f32, seed: f32, size: f32) -> f32 {
  let q = p * scale;
  let cell = floor(q);
  var cover = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let id = cell + vec2f(f32(x), f32(y));
      for (var k = 0; k < 2; k++) {
        let h = hash22(id * 1.7 + seed + f32(k) * 13.1);
        // Wind: each leaf swings about its own phase.
        let phase = h.x * TAU + id.y * 0.7;
        let sway = vec2f(sin(ck.time * 0.9 + phase), cos(ck.time * 0.7 + phase * 1.3)) * 0.06 * ck.sway;
        let c = id + h + sway;
        let a = hash21(id + seed * 3.1 + f32(k)) * TAU + 0.25 * ck.sway * sin(ck.time * 1.1 + phase);
        let s = size * (0.7 + 0.5 * hash21(id - seed - f32(k)));
        let l = rot2(a) * (q - c) / s;
        // Vesica-like leaf: an ellipse pinched toward both tips.
        let w = 0.38 * (1.0 - 0.75 * l.x * l.x);
        let d = max(abs(l.y) - max(w, 0.0), abs(l.x) - 1.0);
        cover = max(cover, smoothstep(0.04, -0.04, d * s));
      }
    }
  }
  return cover;
}

fn cookie_leaves(uv: vec2f) -> vec3f {
  // Three depths of canopy, each gently drifting, plus a couple of branches.
  let drift = vec2f(sin(ck.time * 0.21), cos(ck.time * 0.17)) * 0.01 * ck.sway;
  var t = 1.0;
  t *= 1.0 - 0.9 * leaf_layer(uv + drift, 4.0, 1.0, 0.55);
  t *= 1.0 - 0.85 * leaf_layer(uv + drift * 1.6 + 0.37, 6.5, 7.0, 0.5);
  t *= 1.0 - 0.8 * leaf_layer(uv + drift * 2.3 + 0.71, 9.0, 3.0, 0.45);
  let bend = 0.02 * ck.sway * sin(ck.time * 0.5);
  let b1 = sd_segment(uv, vec2f(-0.1, 0.25), vec2f(0.7 + bend, 0.62)) - 0.012;
  let b2 = sd_segment(uv, vec2f(0.45, 0.47), vec2f(0.9, 0.2 + bend)) - 0.007;
  t *= smoothstep(-0.004, 0.004, min(b1, b2));
  // Sunlight through foliage is warm; the gaps go slightly golden.
  return vec3f(t) * mix(vec3f(1.0, 0.92, 0.78), vec3f(1.0), t);
}

fn cookie_blinds(uv: vec2f) -> vec3f {
  // A window with venetian blinds: tilted slats with soft gaps, two cords.
  let frame_d = sd_box(uv - 0.5, vec2f(0.4, 0.44));
  if (frame_d > 0.0) { return vec3f(0.0); }
  let wave = 0.004 * ck.sway * sin(ck.time * 0.8 + uv.x * 4.0);
  let y = (uv.y + wave) * 22.0;
  let f = fract(y);
  var t = smoothstep(0.52, 0.6, f) * smoothstep(1.0, 0.9, f);
  let cords = min(abs(uv.x - 0.32), abs(uv.x - 0.68));
  t *= smoothstep(0.002, 0.005, cords);
  // The mullion behind the blinds.
  t *= smoothstep(0.006, 0.012, abs(uv.x - 0.5));
  return vec3f(t);
}

fn cookie_window(uv: vec2f) -> vec3f {
  // Six panes behind a sheer curtain drifting in from the left.
  let q = uv - 0.5;
  if (sd_box(q, vec2f(0.38, 0.44)) > 0.0) { return vec3f(0.0); }
  let cell = fract(vec2f((q.x + 0.38) / 0.76 * 2.0, (q.y + 0.44) / 0.88 * 3.0));
  let edge = min(cell, 1.0 - cell) * vec2f(0.38, 0.293);
  var t = smoothstep(0.008, 0.014, min(edge.x, edge.y));
  let curtain = 0.22 + 0.03 * sin(uv.y * 7.0 + ck.time * 0.6 * ck.sway) + 0.015 * sin(uv.y * 19.0 - ck.time * 0.9 * ck.sway);
  let folds = 0.55 + 0.25 * sin(uv.x * 90.0 + 4.0 * sin(uv.y * 3.0));
  t *= select(1.0, folds * 0.5, uv.x < curtain);
  return vec3f(t);
}

@fragment
fn fs_cookie(v: FullscreenOut) -> @location(0) vec4f {
  var c = vec3f(1.0);
  switch ck.kind {
    case 1u: { c = cookie_blinds(v.uv); }
    case 2u: { c = cookie_window(v.uv); }
    default: { c = cookie_leaves(v.uv); }
  }
  return vec4f(c, 1.0);
}
