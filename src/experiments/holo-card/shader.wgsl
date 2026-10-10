// ---------------------------------------------------------------------------
// Holographic Card — a foil trading card that tilts toward the pointer.
//
// Fullscreen ray cast against a rotated card plane (no geometry). The card has
// two layers:
//   print: procedural artwork under a glossy laminate (clearcoat GGX + IBL).
//   foil:  a metallised layer with
//          - a diffraction grating (concentric grooves around the "sun", plus
//            randomly oriented glitter cells) -> rainbow streaks that sweep as
//            the card tilts relative to the light,
//          - thin-film interference tinting its reflections (oil-slick colours),
//          - anisotropic GGX, brushed radially, for stretched highlights.
// The backdrop receives the card's soft shadow and a coloured glow.
// Requires (prepended): common.wgsl, noise.wgsl, sdf.wgsl, brdf.wgsl,
// optics.wgsl, fullscreen.wgsl, `Frame`, `IblParams`, ibl.wgsl.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var title_sdf: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;

const SUN = vec2f(0.0, 0.26);

fn card_rot() -> mat3x3f { return mat3x3f(frame.cardX.xyz, frame.cardY.xyz, frame.cardZ.xyz); }

fn sd_title(p: vec2f) -> f32 {
  let rel = p - frame.titleRect.xy;
  let half = frame.titleRect.zw;
  let box = sd_box(rel, half);
  let uv = clamp(vec2f(rel.x / half.x, -rel.y / half.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
  let d = textureSampleLevel(title_sdf, lin, uv, 0.0).r * frame.titleScale;
  return select(d, max(box, d), box > 0.0);
}

// --- Artwork (card units, y up) ----------------------------------------------------

struct Art { albedo: vec3f, foil: f32, ink: f32 };

fn artwork(p: vec2f) -> Art {
  let half = frame.cardHalf;
  // Synthwave sky.
  var c = mix(vec3f(0.04, 0.02, 0.14), vec3f(0.75, 0.16, 0.45), smoothstep(-0.25, 0.75, p.y));
  c = mix(c, vec3f(0.95, 0.45, 0.35), smoothstep(0.05, -0.25, p.y) * 0.6);
  // Striped retro sun.
  let ds = length(p - SUN) - 0.3;
  let stripes = step(0.5, fract((p.y - SUN.y) * 18.0)) * step(p.y, SUN.y - 0.02);
  let sun_col = mix(vec3f(1.0, 0.85, 0.35), vec3f(1.0, 0.3, 0.45), smoothstep(SUN.y + 0.3, SUN.y - 0.3, p.y));
  let sun_mask = smoothstep(0.004, -0.004, ds) * (1.0 - stripes);
  c = mix(c, sun_col, sun_mask);
  // Mountains (two layers).
  let m1 = smoothstep(0.003, -0.003, p.y - (-0.08 + 0.12 * fbm(vec2f(p.x * 2.6, 1.3), 4)));
  c = mix(c, vec3f(0.18, 0.05, 0.25), m1);
  let m2 = smoothstep(0.003, -0.003, p.y - (-0.25 + 0.08 * fbm(vec2f(p.x * 4.0 + 7.0, 3.1), 4)));
  c = mix(c, vec3f(0.07, 0.02, 0.12), m2);
  // Title band and title.
  let band = smoothstep(0.003, -0.003, p.y - (-half.y + 0.3));
  c = mix(c, vec3f(0.02, 0.015, 0.03), band);
  let title = smoothstep(0.002, -0.002, sd_title(p));
  c = mix(c, vec3f(0.95), title);
  // Frame.
  let inner = sd_round_box(p, half - vec2f(0.055), 0.03);
  let frame_mask = smoothstep(-0.002, 0.002, inner);

  var foil = 0.0;
  switch frame.holoMode {
    case 0u: { foil = max(frame_mask, title); }                                 // frame & title
    case 1u: { foil = max(max(frame_mask, title), (1.0 - sun_mask) * 0.65); }   // reverse holo
    default: { foil = max(frame_mask, 0.8 * (1.0 - band) + title * 0.6); }      // full art
  }
  // Ink sits on top of the foil: where art is printed over foil it tints it.
  return Art(c, foil, 1.0 - frame_mask);
}

/// Base grating: concentric grooves around the sun (tangent to circles, so the
/// rainbow fans out radially from it).
fn grating_base(p: vec2f) -> vec2f {
  let r = p - SUN;
  return normalize(vec2f(-r.y, r.x) + vec2f(1e-4, 0.0));
}

/// Glitter: small round flecks, each a tiny grating with its own random
/// orientation. Returns (direction.xy, coverage).
fn glitter(p: vec2f) -> vec3f {
  let q = p * frame.glitterScale;
  let cell = floor(q);
  let rnd = hash22(cell);
  let rnd2 = hash22(cell + 17.0);
  let present = step(1.0 - frame.glitter, rnd.x);
  let centre = 0.3 + 0.4 * rnd2;
  let fleck = smoothstep(0.32, 0.18, length(fract(q) - centre));
  let a = rnd.y * TAU;
  return vec3f(cos(a), sin(a), present * fleck);
}

// --- Shading -------------------------------------------------------------------------

fn shade_card(p: vec2f, world: vec3f, v: vec3f) -> vec3f {
  let rot = card_rot();
  let n = rot * vec3f(0.0, 0.0, 1.0);
  let art = artwork(p);
  let lv = frame.lightPos - world;
  let l = normalize(lv);
  let light = frame.lightColor / dot(lv, lv);

  // Print layer: diffuse ink + glossy laminate.
  let print_m = Material(art.albedo, 0.6, 0.0);
  let coat = Material(vec3f(0.0), 0.08, 0.0);
  var print = light * brdf_direct(print_m, n, v, l) + ibl_irradiance(n) / PI * art.albedo * frame.envIntensity;
  let coat_f = f_schlick(vec3f(0.04), saturate(dot(n, v))).x;
  print += light * brdf_direct(coat, n, v, l) + ibl_specular_radiance(reflect(-v, n), 0.08) * coat_f * frame.envIntensity;

  // Foil layer.
  let t = normalize(rot * vec3f(grating_base(p), 0.0));
  let gl = glitter(p);
  let tg = normalize(rot * vec3f(gl.xy, 0.0));
  // Film thickness drifts slowly across the card (and with a little noise).
  let thickness = frame.filmThickness * (0.75 + 0.5 * fbm(p * 2.5 + 4.0, 3) + 0.15 * length(p - SUN));
  let film = mix(vec3f(1.0), thin_film(saturate(dot(n, v)), thickness, 1.45), frame.filmStrength);
  // Radially brushed metal: tangent along the radius from the sun.
  let radial = normalize(rot * vec3f(normalize(p - SUN + vec2f(1e-4, 0.0)), 0.0));
  let foil_m = Material(vec3f(0.92) * film, frame.foilRoughness, 1.0);
  var foil = light * brdf_direct_aniso(foil_m, n, radial, v, l, frame.anisotropy);
  foil += ibl_specular_radiance(reflect(-v, n), frame.foilRoughness) * film * 0.9 * frame.envIntensity;
  let rainbow = diffraction_grating(l, v, n, t, frame.grooveSpacing, 0.08);
  // Flecks diffract more widely and flash brighter than the base grating.
  let sparkle = diffraction_grating(l, v, n, tg, frame.grooveSpacing, 0.35) * gl.z * 3.0;
  foil += (rainbow + sparkle) * light * frame.holoIntensity;
  // Printed ink partially covers the foil in the artwork area.
  foil = mix(foil, foil * (art.albedo * 1.6 + 0.15), art.ink * frame.inkCover);

  return mix(print, foil, art.foil * frame.foilAmount);
}

fn shade_backdrop(world: vec3f) -> vec3f {
  // Soft studio backdrop.
  let base = mix(vec3f(0.012, 0.010, 0.020), vec3f(0.05, 0.03, 0.07), smoothstep(-2.5, 1.5, world.y));
  // Shadow of the card: where does the segment to the light cross the card plane?
  let rot = card_rot();
  let lv = frame.lightPos - world;
  let o = transpose(rot) * world;
  let d = transpose(rot) * lv;
  var vis = 1.0;
  if (abs(d.z) > 1e-4) {
    let t = -o.z / d.z;
    if (t > 0.0 && t < 1.0) {
      let s = o.xy + d.xy * t;
      let penumbra = 0.02 + frame.lightRadius * t * 2.0;
      vis = smoothstep(-penumbra, penumbra, sd_round_box(s, frame.cardHalf, frame.cardRadius));
    }
  }
  // Coloured bounce from the card onto the wall behind it.
  let glow = exp(-length(world.xy) * 1.6) * frame.glowColor;
  return base * mix(0.35, 1.0, vis) + glow;
}

@fragment
fn fs_main(fv: FullscreenOut) -> @location(0) vec4f {
  let ndc = vec2f(fv.uv.x * 2.0 - 1.0, 1.0 - fv.uv.y * 2.0);
  let far = frame.invViewProj * vec4f(ndc, 1.0, 1.0);
  let dir = normalize(far.xyz / far.w - frame.cameraPos);
  let v = -dir;

  // Backdrop plane z = backdropZ.
  let tb = (frame.backdropZ - frame.cameraPos.z) / dir.z;
  var col = shade_backdrop(frame.cameraPos + dir * tb);

  // Card plane through the origin, rotated.
  let rot = card_rot();
  let o = transpose(rot) * frame.cameraPos;
  let d = transpose(rot) * dir;
  let t = -o.z / d.z;
  if (t > 0.0) {
    let p = (o + d * t).xy;
    let cd = sd_round_box(p, frame.cardHalf, frame.cardRadius);
    let px = t * frame.pixelAngle;
    let coverage = smoothstep(px, -px, cd);
    if (coverage > 0.0) {
      // The back of the card is plain dark stock.
      let facing = dot(rot * vec3f(0.0, 0.0, 1.0), v) > 0.0;
      let card = select(vec3f(0.02), shade_card(p, frame.cameraPos + dir * t, v), facing);
      col = mix(col, card, coverage);
    }
  }
  return vec4f(col, 1.0);
}
