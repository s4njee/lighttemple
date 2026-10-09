// ---------------------------------------------------------------------------
// sdf.wgsl — signed distance primitives and operators (after Inigo Quilez).
// 2D primitives are `sd_*`, 3D ones `sd_*3`. Negative = inside.
// ---------------------------------------------------------------------------

// --- Transforms ------------------------------------------------------------------

fn rot2(a: f32) -> mat2x2f {
  let c = cos(a); let s = sin(a);
  return mat2x2f(c, s, -s, c);
}

// --- 2D primitives -----------------------------------------------------------------

fn sd_circle(p: vec2f, r: f32) -> f32 { return length(p) - r; }

fn sd_box(p: vec2f, half_size: vec2f) -> f32 {
  let d = abs(p) - half_size;
  return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sd_round_box(p: vec2f, half_size: vec2f, r: f32) -> f32 {
  let q = abs(p) - half_size + r;
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn sd_segment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

/// Capsule / pill between a and b.
fn sd_capsule(p: vec2f, a: vec2f, b: vec2f, r: f32) -> f32 { return sd_segment(p, a, b) - r; }

fn sd_ring(p: vec2f, r: f32, thickness: f32) -> f32 { return abs(length(p) - r) - thickness; }

fn sd_equilateral_triangle(p_in: vec2f, r: f32) -> f32 {
  let k = sqrt(3.0);
  var p = vec2f(abs(p_in.x) - r, p_in.y + r / k);
  if (p.x + k * p.y > 0.0) { p = vec2f(p.x - k * p.y, -k * p.x - p.y) / 2.0; }
  p.x -= clamp(p.x, -2.0 * r, 0.0);
  return -length(p) * sign(p.y);
}

// --- 3D primitives -----------------------------------------------------------------

fn sd_sphere3(p: vec3f, r: f32) -> f32 { return length(p) - r; }

fn sd_box3(p: vec3f, half_size: vec3f) -> f32 {
  let q = abs(p) - half_size;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_round_box3(p: vec3f, half_size: vec3f, r: f32) -> f32 {
  return sd_box3(p, half_size - r) - r;
}

/// Torus in the XZ plane: t.x = major radius, t.y = tube radius.
fn sd_torus3(p: vec3f, t: vec2f) -> f32 {
  return length(vec2f(length(p.xz) - t.x, p.y)) - t.y;
}

fn sd_capsule3(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

/// Vertical cylinder centred at the origin.
fn sd_cylinder3(p: vec3f, radius: f32, half_height: f32) -> f32 {
  let d = abs(vec2f(length(p.xz), p.y)) - vec2f(radius, half_height);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

fn sd_plane3(p: vec3f, n: vec3f, h: f32) -> f32 { return dot(p, n) + h; }

/// Extrudes a 2D distance `d2` along z to half-depth `h` (2D text/logos -> 3D).
fn op_extrude(d2: f32, z: f32, h: f32) -> f32 {
  let w = vec2f(d2, abs(z) - h);
  return min(max(w.x, w.y), 0.0) + length(max(w, vec2f(0.0)));
}

// --- Operators (work for 2D and 3D) -------------------------------------------------

fn op_union(a: f32, b: f32) -> f32 { return min(a, b); }
fn op_subtract(a: f32, b: f32) -> f32 { return max(a, -b); }
fn op_intersect(a: f32, b: f32) -> f32 { return max(a, b); }
fn op_round(d: f32, r: f32) -> f32 { return d - r; }
fn op_onion(d: f32, thickness: f32) -> f32 { return abs(d) - thickness; }

/// Polynomial smooth minimum; `k` is the blend radius in distance units.
fn op_smooth_union(a: f32, b: f32, k: f32) -> f32 {
  let h = max(k - abs(a - b), 0.0) / max(k, 1e-6);
  return min(a, b) - h * h * k * 0.25;
}

fn op_smooth_subtract(a: f32, b: f32, k: f32) -> f32 { return -op_smooth_union(-a, b, k); }
fn op_smooth_intersect(a: f32, b: f32, k: f32) -> f32 { return -op_smooth_union(-a, -b, k); }
