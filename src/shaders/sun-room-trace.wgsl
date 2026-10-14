// ---------------------------------------------------------------------------
// sun-room-trace.wgsl — analytic ray casting against the sun room's primitives
// (yaw-rotated boxes and spheres; see src/scene/sun-room.ts). Brute force over
// every primitive: the room has a few dozen, so no acceleration structure.
//
// Requires (prepended): ROOM_PRIM_WGSL (struct Prim) and a binding
//   var<storage, read> prims: array<Prim>;
// ---------------------------------------------------------------------------

struct RoomHit {
  /// Distance along the ray; negative on a miss.
  t: f32,
  n: vec3f,
  id: u32,
};

/// Rotation about +y by the angle whose cosine and sine are c and s.
fn room_rot_y(v: vec3f, c: f32, s: f32) -> vec3f {
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

/// Nearest intersection of the ray with primitive `pr` beyond `tmin`: (t, normal), t < 0 on a miss.
fn room_prim_hit(pr: Prim, o: vec3f, d: vec3f, tmin: f32) -> vec4f {
  let rel = o - pr.center.xyz;
  if (pr.center.w > 0.5) {
    // Sphere.
    let r = pr.half.x;
    let b = dot(rel, d);
    let c = dot(rel, rel) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) { return vec4f(-1.0); }
    let t = -b - sqrt(disc);
    if (t < tmin) { return vec4f(-1.0); }
    return vec4f(t, (rel + d * t) / r);
  }
  // Box: slab test in the box's own (yaw-rotated) frame.
  let cy = cos(pr.half.w);
  let sy = sin(pr.half.w);
  let ol = room_rot_y(rel, cy, -sy);
  let dl = room_rot_y(d, cy, -sy);
  // A zero component would divide by zero; any tiny value gives the right slab answer.
  let inv = 1.0 / select(dl, vec3f(1e-9), abs(dl) < vec3f(1e-9));
  let h = pr.half.xyz;
  let t1 = (-h - ol) * inv;
  let t2 = (h - ol) * inv;
  let tn = min(t1, t2);
  let tf = max(t1, t2);
  let t_near = max(max(tn.x, tn.y), tn.z);
  let t_far = min(min(tf.x, tf.y), tf.z);
  // Rays starting inside a box (t_near < tmin) pass through it: only outer faces count.
  if (t_near > t_far || t_near < tmin) { return vec4f(-1.0); }
  let axis = step(tn.yzx, tn) * step(tn.zxy, tn);
  let nl = -sign(dl) * axis;
  return vec4f(t_near, room_rot_y(nl, cy, sy));
}

/// Closest hit along the ray within (tmin, tmax).
fn room_trace(o: vec3f, d: vec3f, tmin: f32, tmax: f32) -> RoomHit {
  var hit = RoomHit(-1.0, vec3f(0.0), 0u);
  var best = tmax;
  for (var i = 0u; i < arrayLength(&prims); i++) {
    let h = room_prim_hit(prims[i], o, d, tmin);
    if (h.x >= 0.0 && h.x < best) {
      best = h.x;
      hit = RoomHit(h.x, h.yzw, i);
    }
  }
  return hit;
}

/// Is anything in the way within (tmin, tmax)? Stops at the first hit.
fn room_occluded(o: vec3f, d: vec3f, tmin: f32, tmax: f32) -> bool {
  for (var i = 0u; i < arrayLength(&prims); i++) {
    let h = room_prim_hit(prims[i], o, d, tmin);
    if (h.x >= 0.0 && h.x < tmax) { return true; }
  }
  return false;
}

/// Is `p` inside a solid primitive (more than `margin` from its surface)?
fn room_inside_solid(p: vec3f, margin: f32) -> bool {
  for (var i = 0u; i < arrayLength(&prims); i++) {
    let pr = prims[i];
    let rel = p - pr.center.xyz;
    if (pr.center.w > 0.5) {
      if (length(rel) < pr.half.x - margin) { return true; }
    } else {
      let l = room_rot_y(rel, cos(pr.half.w), -sin(pr.half.w));
      if (all(abs(l) < pr.half.xyz - margin)) { return true; }
    }
  }
  return false;
}
