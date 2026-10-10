// ---------------------------------------------------------------------------
// Ace of Spades — a holographic playing card in the void.
//
// One fullscreen pass. Each pixel ray-marches a thin rounded card (an extruded
// rounded-rectangle SDF) centred at the origin; the camera orbits it.
//
//   front: matte white stock with a classic ornate ace: the spade and the
//          corner pips are polished black marble (warped-noise veins) under a
//          holographic laminate (diffraction rainbow + glitter, thin-film sheen).
//          White printed detail inside the spade: keylines, a ring of dots,
//          mirrored scrollwork, an eye-in-triangle emblem in holographic foil on a
//          white plate, and "mk7s" in tiny red type at the centre.
//   back:  black marble with a holographic lattice, frame and spade emblem.
//   edge:  white paper core.
// Lights: three point lights (two in front, one behind) + studio IBL.
// Requires (prepended): common.wgsl, noise.wgsl, sdf.wgsl, brdf.wgsl, optics.wgsl,
// fullscreen.wgsl, `Ace`, `IblParams`, ibl.wgsl.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> u: Ace;
@group(0) @binding(1) var index_sdf: texture_2d<f32>;
@group(0) @binding(2) var mark_sdf: texture_2d<f32>;
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var emblem_tex: texture_2d<f32>;

// Coverage of the red "mk7s" mark at this pixel. It is written to alpha so the post
// chain composites its background colour (pure #FF0000) there after tonemapping,
// keeping the red exact instead of lit, tonemapped and pastel.
var<private> red_ink: f32 = 0.0;

const SPADE_CENTRE = vec2f(0.0, 0.0);
const SPADE_SIZE = 0.44;

// --- Shapes (card units: x right, y up, card half-size ≈ 0.63 × 0.88) ----------------

fn dot2(v: vec2f) -> f32 { return dot(v, v); }

/// Inigo Quilez's exact heart: point at the origin, lobes up to y ≈ 1.1.
fn sd_heart(p_in: vec2f) -> f32 {
  let p = vec2f(abs(p_in.x), p_in.y);
  if (p.y + p.x > 1.0) { return sqrt(dot2(p - vec2f(0.25, 0.75))) - sqrt(2.0) / 4.0; }
  return sqrt(min(dot2(p - vec2f(0.0, 1.0)), dot2(p - 0.5 * max(p.x + p.y, 0.0)))) * sign(p.x - p.y);
}

/// A classic spade in unit space (y from ≈ -0.75 to 0.85): an upside-down heart
/// for the body, plus a flared stem carved by two large circles.
fn sd_spade_unit(p: vec2f) -> f32 {
  let body = sd_heart(vec2f(p.x, 0.85 - p.y));
  // Reaches up into the body so it also fills the notch between the lobes.
  let stem_box = sd_box(p - vec2f(0.0, -0.37), vec2f(0.24, 0.38));
  let carve = min(length(p - vec2f(0.97, -0.19)), length(p - vec2f(-0.97, -0.19))) - 0.92;
  let stem = max(stem_box, -carve);
  return op_smooth_union(body, stem, 0.04);
}

fn sd_spade(p: vec2f, centre: vec2f, size: f32) -> f32 {
  return sd_spade_unit((p - centre) / size) * size;
}

/// Text from a runtime SDF texture placed at rect = (centre, half-size), card units.
fn sd_text_tex(tex: texture_2d<f32>, rect: vec4f, scale_pad: vec2f, p: vec2f) -> f32 {
  let rel = p - rect.xy;
  let half = rect.zw;
  let box = sd_box(rel, half);
  let uv = clamp(vec2f(rel.x / half.x, -rel.y / half.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
  let d = textureSampleLevel(tex, lin, uv, 0.0).r * scale_pad.x;
  let glyph_box = sd_box(rel, max(half - vec2f(scale_pad.y), vec2f(1e-4)));
  return select(d, max(glyph_box, d - box), box > 0.0);
}

/// Corner index (a condensed serif "A" over a solid spade) at the top-left; call with -p for the bottom-right.
fn sd_index(p: vec2f) -> f32 {
  let a = sd_text_tex(index_sdf, u.indexRect, u.indexScalePad, p);
  let s = sd_spade(p, u.indexPip, 0.052);
  return min(a, s);
}

fn sd_frame(p: vec2f) -> f32 {
  return abs(sd_round_box(p, u.cardHalf - vec2f(0.045), 0.035)) - 0.0022;
}

// --- Ornate centre spade (spade units: y ≈ -0.75 .. 0.85) ------------------------------

/// Archimedean scroll: a curling white line around `c`, out to radius `rmax`.
fn sd_scroll(q: vec2f, c: vec2f, pitch: f32, rmax: f32, turn: f32) -> f32 {
  let d = q - c;
  let r = length(d);
  let a = atan2(d.y, d.x) * turn;
  let s = (r / pitch - a / TAU);
  let line = abs(fract(s + 0.5) - 0.5) * pitch;
  return max(line - 0.011, r - rmax);
}

/// White printed detail inside the black spade: a keyline, a ring of dots, a second
/// keyline, then mirrored scrollwork (the eye emblem is drawn separately). Returns
/// (white ink distance, inner-field distance), in spade units.
fn spade_ornament(s: vec2f) -> vec2f {
  let d0 = sd_spade_unit(s);
  // Fine keylines just inside the edge.
  var white = abs(d0 + 0.05) - 0.0045;
  white = min(white, abs(d0 + 0.155) - 0.0045);
  // Ring of dots between the keylines, spaced by angle around the body. Where the
  // contour runs toward the centre (the notch above the stem) angle spacing would
  // smear the dots into a rope, so they are left out there, as on real cards.
  let c = vec2f(0.0, 0.12);
  let rel = s - c;
  let e = vec2f(0.004, 0.0);
  let g = normalize(vec2f(sd_spade_unit(s + e.xy) - sd_spade_unit(s - e.xy), sd_spade_unit(s + e.yx) - sd_spade_unit(s - e.yx)) + 1e-6);
  let facing = abs(dot(g, normalize(rel + 1e-6)));
  let ang = atan2(rel.x, rel.y);
  let count = 52.0;
  let cell = (fract(ang / TAU * count + 0.5) - 0.5) * TAU / count * length(rel);
  let dots_ok = step(0.72, facing) * step(-0.36, s.y);
  white = min(white, mix(1e3, length(vec2f(d0 + 0.1025, cell)) - 0.019, dots_ok));

  // Scrollwork, mirrored left/right, kept inside the inner keyline.
  let m = vec2f(abs(s.x), s.y);
  let inner = d0 + 0.19;
  var orn = sd_scroll(m, vec2f(0.33, -0.02), 0.05, 0.12, 1.0) + 0.0035;
  orn = min(orn, sd_scroll(m, vec2f(0.19, 0.42), 0.04, 0.085, -1.0) + 0.0035);
  orn = max(orn, inner);
  // Clear black space for the tiny "mk7s" mark under the emblem.
  orn = max(orn, -sd_round_box(s - vec2f(0.0, 0.045), vec2f(0.15, 0.04), 0.02));
  white = min(white, orn);
  return vec2f(white, inner);
}

// --- Eye-in-triangle emblem (replaces the club) ---------------------------------------
// An inverted engraving (eye-emblem.png): R = stroke coverage (drawn in holographic
// foil), G = the triangular white plate it sits on. Placed at rect = (centre, half-size).

/// Returns (plate coverage, foil coverage) at card point p; `aa` = pixel size.
fn emblem(p: vec2f, aa: f32) -> vec2f {
  let rel = (p - u.emblemRect.xy) / u.emblemRect.zw;
  if (any(abs(rel) > vec2f(1.0))) { return vec2f(0.0); }
  let uv = vec2f(rel.x, -rel.y) * 0.5 + 0.5;
  // Explicit LOD from the pixel footprint (shading runs after a non-uniform loop).
  let lod = log2(max(aa / u.emblemTexel, 1.0));
  let t = textureSampleLevel(emblem_tex, lin, uv, lod);
  let plate = smoothstep(0.3, 0.7, t.g);
  return vec2f(plate, saturate(t.r * 1.15) * plate);
}

/// Polished black marble: near-black stone with warped grey-white veins.
fn marble(p: vec2f) -> vec3f {
  let q = p * 2.2 + vec2f(3.1, 1.7);
  let warp = fbm(q * 1.3, 4);
  let v = fbm(q * 1.8 + vec2f(warp * 1.6, warp * 0.9), 5);
  let vein = smoothstep(0.03, 0.0, abs(v + 0.05));
  let fine = smoothstep(0.012, 0.0, abs(fbm(q * 5.0 + warp * 2.5, 4))) * 0.45;
  let cloud = 0.5 + 0.5 * fbm(q * 0.9 + 7.0, 3);
  let base = mix(vec3f(0.006, 0.006, 0.008), vec3f(0.02, 0.019, 0.022), cloud);
  return mix(base, vec3f(0.16, 0.157, 0.167), saturate(vein * 0.7 + fine * 0.35));
}

// --- Card body ---------------------------------------------------------------------------

fn sd_card(q: vec3f) -> f32 {
  return op_extrude(sd_round_box(q.xy, u.cardHalf, u.cardRadius), q.z, u.halfThickness);
}

fn card_normal(q: vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.0004;
  return normalize(e.xyy * sd_card(q + e.xyy) + e.yyx * sd_card(q + e.yyx) + e.yxy * sd_card(q + e.yxy) + e.xxx * sd_card(q + e.xxx));
}

// --- Lighting ------------------------------------------------------------------------------

struct Light { pos: vec3f, color: vec3f };

fn light(i: i32) -> Light {
  switch i {
    case 0: { return Light(u.light0.xyz, u.light0Color.rgb); }
    case 1: { return Light(u.light1.xyz, u.light1Color.rgb); }
    default: { return Light(u.light2.xyz, u.light2Color.rgb); }
  }
}

/// Glitter: small round flecks with random groove directions. (direction.xy, coverage)
fn glitter(p: vec2f) -> vec3f {
  let q = p * u.glitterScale;
  let cell = floor(q);
  let rnd = hash22(cell);
  let rnd2 = hash22(cell + 17.0);
  let present = step(1.0 - u.glitter, rnd.x);
  let fleck = smoothstep(0.32, 0.18, length(fract(q) - (0.3 + 0.4 * rnd2)));
  let a = rnd.y * TAU;
  return vec3f(cos(a), sin(a), present * fleck);
}

/// Holographic laminate: diffraction rainbow (grooves concentric around `centre`)
/// plus glitter flecks, from every light. Added on top of whatever is underneath.
fn holo(p: vec2f, centre: vec2f, pos: vec3f, n: vec3f, v: vec3f, ax: vec3f, ay: vec3f) -> vec3f {
  let r = p - centre + vec2f(1e-4, 0.0);
  let tangent2 = normalize(vec2f(-r.y, r.x));
  let t = normalize(ax * tangent2.x + ay * tangent2.y);
  let gl = glitter(p);
  let tg = normalize(ax * gl.x + ay * gl.y);
  var c = vec3f(0.0);
  for (var i = 0; i < 3; i++) {
    let lt = light(i);
    let lv = lt.pos - pos;
    let l = normalize(lv);
    let li = lt.color / dot(lv, lv);
    let rainbow = diffraction_grating(l, v, n, t, u.grooveSpacing, 0.08);
    let sparkle = diffraction_grating(l, v, n, tg, u.grooveSpacing, 0.35) * gl.z * 3.0;
    c += (rainbow + sparkle) * li;
  }
  return c * u.holoIntensity;
}

/// Silver holographic foil: metallic, thin-film tinted, rainbow laminate on top.
fn shade_holofoil(p: vec2f, centre: vec2f, pos: vec3f, n: vec3f, v: vec3f, ax: vec3f, ay: vec3f) -> vec3f {
  let film_t = u.filmThickness * (0.75 + 0.5 * fbm(p * 6.0 + 1.0, 3));
  let film = mix(vec3f(1.0), thin_film(saturate(dot(n, v)), film_t, 1.45), u.filmStrength);
  let m = Material(vec3f(0.9) * film, 0.18, 1.0);
  var c = ibl_specular_radiance(reflect(-v, n), 0.18) * film * u.envIntensity;
  for (var i = 0; i < 3; i++) {
    let lt = light(i);
    let lv = lt.pos - pos;
    c += lt.color / dot(lv, lv) * brdf_direct(m, n, v, normalize(lv));
  }
  return c + holo(p, centre, pos, n, v, ax, ay) * 1.5;
}

/// Holographic black marble: polished stone with an iridescent film and rainbow laminate.
fn shade_marble(p: vec2f, centre: vec2f, pos: vec3f, n: vec3f, v: vec3f, ax: vec3f, ay: vec3f) -> vec3f {
  let film_t = u.filmThickness * (0.75 + 0.5 * fbm(p * 3.0 + 4.0, 3));
  let film = mix(vec3f(1.0), thin_film(saturate(dot(n, v)), film_t, 1.45), u.filmStrength);
  let albedo = marble(p);
  let stone = Material(albedo, 0.35, 0.0);
  let gloss = Material(vec3f(0.0), u.foilRoughness, 0.0);
  let coat_f = f_schlick(vec3f(0.05), saturate(dot(n, v))).x;
  var c = ibl_irradiance(n) / PI * albedo * u.envIntensity;
  c += ibl_specular_radiance(reflect(-v, n), u.foilRoughness) * coat_f * film * u.envIntensity * 1.6;
  for (var i = 0; i < 3; i++) {
    let lt = light(i);
    let lv = lt.pos - pos;
    let l = normalize(lv);
    let li = lt.color / dot(lv, lv);
    c += li * (brdf_direct(stone, n, v, l) + brdf_direct(gloss, n, v, l) * film * 1.4);
  }
  return c + holo(p, centre, pos, n, v, ax, ay);
}

/// Matte uncoated stock / ink: diffuse with a very rough, faint specular, so the
/// white card reads as paper and only the holographic layers catch highlights.
fn shade_matte(albedo: vec3f, pos: vec3f, n: vec3f, v: vec3f) -> vec3f {
  let base = Material(albedo, 0.95, 0.0);
  var c = ibl_irradiance(n) / PI * albedo * u.envIntensity;
  for (var i = 0; i < 3; i++) {
    let lt = light(i);
    let lv = lt.pos - pos;
    c += lt.color / dot(lv, lv) * brdf_direct(base, n, v, normalize(lv));
  }
  return c;
}

fn shade_front(p: vec2f, pos: vec3f, n: vec3f, v: vec3f, aa: f32) -> vec3f {
  let ax = vec3f(1.0, 0.0, 0.0);
  let ay = vec3f(0.0, 1.0, 0.0);
  let s = (p - SPADE_CENTRE) / SPADE_SIZE;
  let spade = sd_spade_unit(s) * SPADE_SIZE;
  let marble_d = min(spade, min(sd_index(p), sd_index(-p)));
  let stone = smoothstep(aa, -aa, marble_d);

  var c = shade_matte(vec3f(0.97), pos, n, v);
  if (stone > 0.0) {
    var black = shade_marble(p, SPADE_CENTRE + vec2f(0.0, 0.05), pos, n, v, ax, ay);
    // White printed ornament and the tiny "mk7s" mark on the centre spade.
    if (spade < aa) {
      let orn = spade_ornament(s);
      let white_ink = smoothstep(aa, -aa, orn.x * SPADE_SIZE);
      if (white_ink > 0.0) { black = mix(black, shade_matte(vec3f(0.95), pos, n, v), white_ink); }
      // Eye-in-triangle emblem: white plate with holographic-foil engraving.
      let em = emblem(p, aa);
      if (em.x > 0.0) {
        black = mix(black, shade_matte(vec3f(0.95), pos, n, v), em.x);
        if (em.y > 0.0) { black = mix(black, shade_holofoil(p, u.emblemRect.xy, pos, n, v, ax, ay), em.y); }
      }
      // "mk7s": flat pure red, composited in post (see `red_ink`).
      red_ink = stone * smoothstep(aa, -aa, sd_text_tex(mark_sdf, u.markRect, u.markScalePad, p));
    }
    c = mix(c, black, stone);
  }
  return c;
}

fn shade_back(p: vec2f, pos: vec3f, n: vec3f, v: vec3f, aa: f32) -> vec3f {
  // Seen from behind, the card's local x axis points along -world x.
  let ax = vec3f(-1.0, 0.0, 0.0);
  let ay = vec3f(0.0, 1.0, 0.0);
  let inner = sd_round_box(p, u.cardHalf - vec2f(0.075), 0.03);
  let q = rot2(0.7853982) * p * 22.0;
  let lattice = min(abs(fract(q.x) - 0.5), abs(fract(q.y) - 0.5)) / 22.0 - 0.0013;
  let badge = sd_spade(p, vec2f(0.0, 0.0), 0.16);
  let ring = abs(length(p) - 0.2) - 0.004;
  var line_d = min(sd_frame(p), max(lattice, inner));
  line_d = max(line_d, -(length(p) - 0.24));
  line_d = min(line_d, min(badge, ring));
  let lines = smoothstep(aa, -aa, line_d);
  var c = shade_marble(p, vec2f(0.0), pos, n, v, ax, ay);
  if (lines > 0.0) {
    c = mix(c, shade_matte(vec3f(0.75, 0.74, 0.72), pos, n, v) + holo(p, vec2f(0.0), pos, n, v, ax, ay), lines);
  }
  return c;
}

fn shade_card(pos: vec3f, v: vec3f, footprint: f32) -> vec3f {
  let n = card_normal(pos);
  // Pixel footprint on the surface grows at grazing angles.
  let aa = footprint / max(abs(n.z) * 0.5 + 0.5 * abs(dot(n, v)), 0.15);
  if (n.z > 0.7) { return shade_front(pos.xy, pos, n, v, aa); }
  if (n.z < -0.7) { return shade_back(vec2f(-pos.x, pos.y), pos, n, v, aa); }
  // Edge: white paper core.
  return shade_matte(vec3f(0.8, 0.79, 0.77), pos, n, v);
}

@fragment
fn fs_main(fv: FullscreenOut) -> @location(0) vec4f {
  let ndc = vec2f(fv.uv.x * 2.0 - 1.0, 1.0 - fv.uv.y * 2.0);
  let far = u.invViewProj * vec4f(ndc, 1.0, 1.0);
  let ro = u.cameraPos;
  let rd = normalize(far.xyz / far.w - ro);

  // Bounding sphere around the card.
  let b = dot(ro, rd);
  let disc = b * b - (dot(ro, ro) - 1.2);
  if (disc < 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  var t = max(-b - sqrt(disc), 0.0);
  let t_end = -b + sqrt(disc);

  var hit = false;
  var closest = 1e9;
  var closest_t = t;
  for (var i = 0; i < 96; i++) {
    let d = sd_card(ro + rd * t);
    if (d < closest) {
      closest = d;
      closest_t = t;
    }
    if (d < 1e-4 * (1.0 + t)) {
      hit = true;
      break;
    }
    t += d;
    if (t > t_end) { break; }
  }

  // Anti-aliased silhouette: near misses become partial coverage.
  let footprint = closest_t * u.pixelAngle;
  var coverage = select(saturate(1.0 - closest / footprint), 1.0, hit);
  if (coverage <= 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let pos = ro + rd * select(closest_t, t, hit);
  let col = shade_card(pos, -rd, footprint);
  // Premultiplied: alpha < 1 lets the post background (pure red) show through the mark.
  let a = 1.0 - red_ink * coverage;
  return vec4f(col * coverage * a, a);
}
