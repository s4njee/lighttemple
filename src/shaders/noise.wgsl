// ---------------------------------------------------------------------------
// noise.wgsl — procedural noise for materials and motion.
//   hash22 / value_noise / gradient_noise / fbm / worley (cellular, F1 & F2)
// All 2D, all deterministic (no textures), periodic only by accident.
// ---------------------------------------------------------------------------

fn hash22(p: vec2f) -> vec2f {
  var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

fn hash21(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

/// Value noise in [0, 1], quintic interpolation.
fn value_noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/// Gradient (Perlin-style) noise in roughly [-1, 1].
fn gradient_noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let g00 = hash22(i) * 2.0 - 1.0;
  let g10 = hash22(i + vec2f(1.0, 0.0)) * 2.0 - 1.0;
  let g01 = hash22(i + vec2f(0.0, 1.0)) * 2.0 - 1.0;
  let g11 = hash22(i + vec2f(1.0, 1.0)) * 2.0 - 1.0;
  let n00 = dot(g00, f);
  let n10 = dot(g10, f - vec2f(1.0, 0.0));
  let n01 = dot(g01, f - vec2f(0.0, 1.0));
  let n11 = dot(g11, f - vec2f(1.0, 1.0));
  return 1.4 * mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
}

/// Fractal sum of gradient noise, ~[-1, 1]. `octaves` <= 8.
fn fbm(p_in: vec2f, octaves: i32) -> f32 {
  var p = p_in;
  var a = 0.5;
  var sum = 0.0;
  // Rotate between octaves to hide grid alignment.
  let r = mat2x2f(0.8, 0.6, -0.6, 0.8);
  for (var i = 0; i < octaves; i++) {
    sum += a * gradient_noise(p);
    p = r * p * 2.03 + vec2f(17.1, 9.2);
    a *= 0.5;
  }
  return sum;
}

/// Cellular noise: distances to the nearest (x) and second-nearest (y) feature points.
fn worley(p: vec2f) -> vec2f {
  let i = floor(p);
  let f = fract(p);
  var d1 = 8.0;
  var d2 = 8.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let g = vec2f(f32(x), f32(y));
      let o = hash22(i + g);
      let d = length(g + o - f);
      if (d < d1) {
        d2 = d1;
        d1 = d;
      } else if (d < d2) {
        d2 = d;
      }
    }
  }
  return vec2f(d1, d2);
}
