// ---------------------------------------------------------------------------
// sun-room.wgsl — surface detail for the shared sun room (src/scene/sun-room.ts).
//
// `room_albedo` modulates a primitive's flat albedo with a procedural pattern
// whose average is about 1, so light baked against the flat colours (probes,
// RSM flux) still matches what the camera sees.
//
// Requires (prepended): common.wgsl, noise.wgsl.
// ---------------------------------------------------------------------------

const ROOM_MAT_FLOOR = 1u;
const ROOM_MAT_WALL = 2u;
const ROOM_MAT_CEILING = 3u;
const ROOM_MAT_RUG = 4u;
const ROOM_MAT_WOOD = 5u;
const ROOM_MAT_FABRIC = 6u;
const ROOM_MAT_LAMP = 7u;
const ROOM_MAT_ART = 8u;
const ROOM_MAT_SKY = 9u;

/// Interior bounds and the window opening (z0, z1, y0, y1), as built in sun-room.ts.
const ROOM_MIN = vec3f(-3.0, 0.0, -2.5);
const ROOM_MAX = vec3f(3.0, 3.0, 2.5);
const ROOM_WINDOW = vec4f(-1.5, 0.9, 0.55, 2.45);
/// Rug centre (xz) and half extents.
const ROOM_RUG = vec4f(-1.2, -0.9, 1.2, 0.95);
/// Print above the sofa: centre (xy) and half extents.
const ROOM_ART = vec4f(1.3, 1.78, 0.55, 0.36);

/// Flat dark "poché" for the cut edges and outer faces of the shell, as on an architect's section.
const ROOM_CUT_COLOR = vec3f(0.03, 0.026, 0.023);

/// Is p on the shell's cut edges or outside faces, rather than a surface inside the room?
/// (The window's reveals count as inside.)
fn room_is_cut(p: vec3f) -> bool {
  let outside = any(p < ROOM_MIN - 0.01) || any(p > ROOM_MAX + 0.01);
  let w = ROOM_WINDOW + vec4f(-0.01, 0.01, -0.01, 0.01);
  let reveal = p.x < ROOM_MIN.x && p.z > w.x && p.z < w.y && p.y > w.z && p.y < w.w;
  return outside && !reveal;
}

/// The view out of the window along d: a sky gradient above the horizon, sunlit lawn below.
/// (No sun disc: direct sunlight is the shadow map's job.) Colours from `outdoorColors()` in sun-room.ts.
fn room_outdoor(d: vec3f, zenith: vec3f, horizon: vec3f, ground: vec3f) -> vec3f {
  if (d.y < 0.0) { return ground; }
  return mix(horizon, zenith, sqrt(d.y));
}

/// A lookup point moved back inside the room (with a small margin), e.g. after a view-dependent bias.
fn room_clamp_inside(p: vec3f, margin: f32) -> vec3f {
  return clamp(p, ROOM_MIN + margin, ROOM_MAX - margin);
}

fn room_albedo(mat: u32, base: vec3f, p: vec3f, n: vec3f) -> vec3f {
  switch mat {
    case ROOM_MAT_FLOOR: {
      if (n.y < 0.5) { return base * 0.7; }
      // Oak planks along x: staggered ends, per-plank tone, grain and dark seams.
      let w = 0.18;
      let row = floor(p.z / w);
      let len = 1.5;
      let shift = hash21(vec2f(row, 3.1));
      let id = vec2f(row, floor(p.x / len + shift));
      let tone = 0.85 + 0.3 * hash21(id);
      let grain = 1.0 + 0.12 * fbm(vec2f(p.x * 1.5, p.z * 30.0) + id * 7.0, 3);
      let fz = fract(p.z / w);
      let fx = fract(p.x / len + shift);
      let seam = min(min(fz, 1.0 - fz) * w, min(fx, 1.0 - fx) * len);
      return base * tone * grain * mix(0.55, 1.0, smoothstep(0.0, 0.004, seam));
    }
    case ROOM_MAT_RUG: {
      // A border band and a lattice in a lighter tone, over a woolly field.
      let q = abs(p.xz - ROOM_RUG.xy) / ROOM_RUG.zw;
      let edge = max(q.x, q.y);
      var c = base;
      let band = vec3f(0.62, 0.45, 0.28);
      if (edge > 0.86) { c = base * 0.75; }
      if (edge > 0.8 && edge < 0.84) { c = band; }
      let d = abs(fract(vec2f(q.x * 3.0 + q.y * 3.0, q.x * 3.0 - q.y * 3.0)) - 0.5);
      if (edge < 0.74 && min(d.x, d.y) < 0.035) { c = mix(base, band, 0.6); }
      return c * (0.9 + 0.2 * value_noise(p.xz * 160.0));
    }
    case ROOM_MAT_WALL, ROOM_MAT_CEILING: {
      // Limewash: a faint, large-scale mottle.
      let along = select(p.x, p.z, abs(n.x) > 0.5);
      let uv = select(vec2f(along, p.y), p.xz, abs(n.y) > 0.5);
      return base * (0.96 + 0.06 * fbm(uv * 2.5, 3));
    }
    case ROOM_MAT_WOOD: {
      let uv = select(p.xz, select(p.zy, p.xy, abs(n.z) > 0.5), abs(n.y) < 0.5);
      return base * (0.92 + 0.16 * fbm(vec2f(uv.x * 2.0, uv.y * 24.0), 3));
    }
    case ROOM_MAT_FABRIC: {
      let uv = select(select(p.zy, p.xy, abs(n.z) > 0.5), p.xz, abs(n.y) > 0.5);
      return base * (0.94 + 0.12 * value_noise(uv * 40.0));
    }
    case ROOM_MAT_ART: {
      if (n.z < 0.5) { return vec3f(0.08); }
      // A simple abstract print: a sun disc over two bands, inside a white mount.
      let q = (p.xy - ROOM_ART.xy) / ROOM_ART.zw;
      if (max(abs(q.x), abs(q.y)) > 0.86) { return vec3f(0.8, 0.78, 0.74); }
      var c = select(vec3f(0.32, 0.42, 0.42), base, q.y > -0.2);
      if (q.y < -0.45) { c = vec3f(0.12, 0.2, 0.22); }
      if (length(q - vec2f(0.25, 0.25)) < 0.3) { c = vec3f(0.85, 0.32, 0.18); }
      return c;
    }
    default: { return base; }
  }
}
