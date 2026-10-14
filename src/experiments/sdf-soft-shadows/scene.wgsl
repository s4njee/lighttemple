// ---------------------------------------------------------------------------
// The sculpture garden as a signed distance field. Shared by the main ray
// marcher, the light-space depth pass (for the PCSS comparison) and the bake.
// Requires (prepended): sdf.wgsl, the `Frame` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;

const MAT_FLOOR = 0.0;
const MAT_STONE = 1.0;
const MAT_COPPER = 2.0;
const MAT_CLAY = 3.0;
const MAT_STEEL = 4.0;
const MAT_SLATE = 5.0;
const MAT_PEARL = 6.0;

const BLOB_CENTER = vec3f(-0.15, 0.0, -1.35);

/// Keeps the closer of two (distance, material) pairs.
fn closer(a: vec2f, d: f32, m: f32) -> vec2f {
  return select(a, vec2f(d, m), d < a.x);
}

fn rot_y(p: vec3f, a: f32) -> vec3f {
  let c = cos(a); let s = sin(a);
  return vec3f(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

/// A clay figure: six spheres melted together with smooth unions. Easily the
/// most expensive thing in the scene, and exactly what a bake makes free.
fn sd_blob(p_world: vec3f) -> f32 {
  let p = p_world - BLOB_CENTER;
  // Bounding sphere: far away, its distance is a valid (smaller) bound.
  let bound = length(p - vec3f(0.0, 0.6, 0.0)) - 1.0;
  if (bound > 0.3) { return bound; }
  var d = sd_sphere3(p - vec3f(0.0, 0.33, 0.0), 0.36);
  d = op_smooth_union(d, sd_sphere3(p - vec3f(0.34, 0.22, 0.16), 0.24), 0.25);
  d = op_smooth_union(d, sd_sphere3(p - vec3f(-0.3, 0.2, 0.2), 0.21), 0.25);
  d = op_smooth_union(d, sd_sphere3(p - vec3f(0.08, 0.74, -0.05), 0.24), 0.28);
  d = op_smooth_union(d, sd_sphere3(p - vec3f(-0.14, 1.0, 0.06), 0.15), 0.22);
  d = op_smooth_union(d, sd_sphere3(p - vec3f(0.18, 1.14, 0.0), 0.11), 0.2);
  return d;
}

/// Everything that never moves (and so can be baked). The floor is left out:
/// an infinite plane doesn't fit in a volume, and it is free to evaluate anyway.
fn map_static(p: vec3f) -> vec2f {
  var res = vec2f(1e9, MAT_STONE);

  // An arch: a slab with a round-headed doorway.
  {
    let q = p - vec3f(-1.75, 0.0, -0.55);
    let slab = sd_box3(q - vec3f(0.0, 1.05, 0.0), vec3f(0.75, 1.05, 0.16));
    let door = min(sd_box3(q - vec3f(0.0, 0.5, 0.0), vec3f(0.36, 0.5, 0.3)), sd_cylinder3(vec3f(q.x, q.z, q.y - 1.0), 0.36, 0.3));
    res = closer(res, max(slab, -door), MAT_STONE);
  }
  // A copper ring standing on its edge.
  {
    let q = rot_y(p - vec3f(0.45, 0.62, 0.1), 0.65);
    res = closer(res, sd_torus3(vec3f(q.x, q.z, q.y), vec2f(0.5, 0.11)), MAT_COPPER);
  }
  res = closer(res, sd_blob(p), MAT_CLAY);
  // A thin pole: sharp shadow at its foot, soft at its tip (contact hardening).
  res = closer(res, sd_capsule3(p, vec3f(1.3, 0.0, -1.1), vec3f(1.3, 2.3, -1.1), 0.035), MAT_STEEL);
  // A rounded block.
  {
    let q = rot_y(p - vec3f(-0.55, 0.32, 1.15), 0.6);
    res = closer(res, sd_round_box3(q, vec3f(0.32, 0.32, 0.32), 0.05), MAT_SLATE);
  }
  return res;
}

/// The one moving object: a pearl bobbing above the floor (always analytic).
fn map_dynamic(p: vec3f) -> f32 {
  return sd_sphere3(p - vec3f(1.5, frame.bobY, 0.95), 0.42);
}

fn map(p: vec3f) -> vec2f {
  var res = closer(map_static(p), map_dynamic(p), MAT_PEARL);
  return closer(res, p.y, MAT_FLOOR);
}
