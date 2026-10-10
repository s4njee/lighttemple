// ---------------------------------------------------------------------------
// Long Shadows — flat-design "long shadow" icons, done with SDF sweeps.
//
// A long shadow is the shape swept along the light direction: a point p is in
// shadow if some p - dir·t (0 <= t <= L) lies inside the shape. We sphere-trace
// that backwards: d = sdf(p - dir·t) is a safe step, so a handful of steps
// cover hundreds of pixels. The hit distance t also gives a fade-out, and the
// clearance gives a little anti-aliasing at the shadow's edges.
//
// Two levels, like classic flat icons: tiles cast shadows onto the page, and
// each tile's glyph casts a shadow that stays inside its tile.
// Requires (prepended): common.wgsl, sdf.wgsl, fullscreen.wgsl, `Flat`.
// ---------------------------------------------------------------------------

struct Tile {
  // centre.xy, half size.zw (px)
  rect: vec4f,
  // glyph region in the text SDF texture (u0, v0, u1, v1)
  glyph_uv: vec4f,
  // rgb colour, a = corner radius (px)
  color: vec4f,
  // x = px per text texel, yz = offset (texels) from the region centre to the glyph's visual centre
  glyph: vec4f,
};

@group(0) @binding(0) var<uniform> u: Flat;
@group(0) @binding(1) var<storage, read> tiles: array<Tile>;
@group(0) @binding(2) var text_sdf: texture_2d<f32>;
@group(0) @binding(3) var lin: sampler;

fn sd_tile(p: vec2f, i: u32) -> f32 {
  let t = tiles[i];
  let r = t.color.a;
  return mix(sd_round_box(p - t.rect.xy, t.rect.zw, r), length(p - t.rect.xy) - t.rect.z, u.circles);
}

fn sd_tiles(p: vec2f) -> f32 {
  var d = 1e9;
  for (var i = 0u; i < u.count; i++) { d = min(d, sd_tile(p, i)); }
  return d;
}

/// Glyph of tile i, from its slice of the text SDF (px).
fn sd_glyph(p: vec2f, i: u32) -> f32 {
  let t = tiles[i];
  let dims = vec2f(textureDimensions(text_sdf));
  let texel = (p - t.rect.xy) / t.glyph.x; // texels from the tile centre
  let centre = vec2f(0.5 * (t.glyph_uv.x + t.glyph_uv.z), 0.5 * (t.glyph_uv.y + t.glyph_uv.w)) * dims;
  let tp = centre + texel + t.glyph.yz;
  let lo = t.glyph_uv.xy * dims;
  let hi = t.glyph_uv.zw * dims;
  let box = sd_box(tp - 0.5 * (lo + hi), 0.5 * (hi - lo));
  let d = textureSampleLevel(text_sdf, lin, clamp(tp, lo, hi) / dims, 0.0).r;
  return max(d, box) * t.glyph.x;
}

/// Exact long shadow of a convex tile: distance from p to the tile swept along
/// dir over [0, L]. The distance from points on a line to a convex set is a
/// convex function, so a ternary search finds the minimum in a few steps.
/// Returns (swept distance, t at the minimum).
fn sweep_tile(p: vec2f, i: u32) -> vec2f {
  var lo = 0.0;
  var hi = u.length;
  for (var k = 0; k < 22; k++) {
    let m1 = lo + (hi - lo) / 3.0;
    let m2 = hi - (hi - lo) / 3.0;
    if (sd_tile(p - u.dir * m1, i) < sd_tile(p - u.dir * m2, i)) { hi = m2; } else { lo = m1; }
  }
  let t = 0.5 * (lo + hi);
  return vec2f(sd_tile(p - u.dir * t, i), t);
}

/// Long-shadow coverage of the page at p, 0 (lit) .. 1 (full shadow).
fn sweep_tiles(p: vec2f) -> f32 {
  var best = 1e9;
  var best_t = 0.0;
  for (var i = 0u; i < u.count; i++) {
    let r = tiles[i].rect;
    // Cheap reject: outside the capsule bounding the whole sweep.
    if (sd_capsule(p, r.xy, r.xy + u.dir * u.length, length(r.zw)) > 1.0) { continue; }
    let s = sweep_tile(p, i);
    if (s.x < best) {
      best = s.x;
      best_t = s.y;
    }
  }
  return smoothstep(0.75, -0.75, best) * (1.0 - u.fade * best_t / u.length);
}

/// Glyph shadows (non-convex, short): dense uniform samples along the sweep.
/// Each sample's coverage is softened by half a step, so the union of samples
/// is continuous and anti-aliased (sphere tracing streaks at grazing angles here).
fn sweep_glyph(p: vec2f, i: u32) -> f32 {
  let limit = 2.0 * length(tiles[i].rect.zw); // never longer than the tile
  let n = 48.0;
  let step = limit / n;
  var cover = 0.0;
  for (var k = 1.0; k <= n; k += 1.0) {
    let t = k * step;
    let c = smoothstep(0.6 * step, -0.6 * step, sd_glyph(p - u.dir * t, i)) * (1.0 - u.fade * t / u.length);
    cover = max(cover, c);
  }
  return cover;
}

@fragment
fn fs_main(v: FullscreenOut) -> @location(0) vec4f {
  let p = v.position.xy;
  // Page with its long shadows (multiplied, like a darker flat tint).
  var col = u.background;
  col = mix(col, col * u.shadowTint, sweep_tiles(p) * u.shadowStrength);

  for (var i = 0u; i < u.count; i++) {
    let d = sd_tile(p, i);
    let cover = smoothstep(0.75, -0.75, d);
    if (cover <= 0.0) { continue; }
    var c = tiles[i].color.rgb;
    c = mix(c, c * u.shadowTint, sweep_glyph(p, i) * u.shadowStrength);
    let g = smoothstep(0.75, -0.75, sd_glyph(p, i));
    c = mix(c, u.glyphColor, g);
    col = mix(col, c, cover);
  }
  return vec4f(col, 1.0);
}
