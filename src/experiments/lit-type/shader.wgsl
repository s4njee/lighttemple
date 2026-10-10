// ---------------------------------------------------------------------------
// Lit Typography — real DOM text, rendered as lit, extruded 3D letters.
//
// Units are canvas pixels. The wall (the page) is the plane z = 0; letters are
// the text SDF extruded toward the viewer to z = depth, with rounded edges.
//
// Oblique projection: every pixel's ray is p + shift·(depth − z) at height z.
// The letters' FRONT faces (z = depth) therefore land exactly where the DOM
// draws its (transparent) glyphs, so selection and editing line up, while a
// non-zero `shift` (from the pointer) reveals the extruded sides like a 3D tilt.
//
// Per pixel: sphere-trace the extruded SDF + wall, shade with GGX from the
// cursor light (soft SDF shadows, SDF ambient occlusion) plus IBL reflections;
// each letter gets its own material.
// Requires (prepended): common.wgsl, noise.wgsl, sdf.wgsl, brdf.wgsl,
// fullscreen.wgsl, `LitType`, `IblParams`, ibl.wgsl.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> u: LitType;
@group(0) @binding(1) var text_sdf: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var<storage, read> letters: array<vec4f>; // DOM x0, x1 (px), x shift (px), hue

/// Index of the letter owning column x (letters are sorted by x): binary search
/// for the last letter starting at or before x, then pick the nearer neighbour.
fn letter_at(x: f32) -> u32 {
  let n = u.letterCount;
  if (n == 0u) { return 0u; }
  var lo = 0u;
  var hi = n - 1u;
  loop {
    if (lo >= hi) { break; }
    let mid = (lo + hi + 1u) / 2u;
    if (letters[mid].x <= x) { lo = mid; } else { hi = mid - 1u; }
  }
  if (lo + 1u < n) {
    let a = letters[lo];
    let b = letters[lo + 1u];
    if (x - a.y > b.x - x) { return lo + 1u; }
  }
  return lo;
}

fn sd_text2(p_in: vec2f) -> f32 {
  // Per-letter horizontal correction so each glyph sits exactly on its DOM glyph.
  let p = vec2f(p_in.x - letters[letter_at(p_in.x)].z, p_in.y);
  let rel = p - u.textRect.xy;
  let half = u.textRect.zw;
  let box = sd_box(rel, half);
  let uv = clamp(rel / half * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
  let d = textureSampleLevel(text_sdf, lin, uv, 0.0).r * u.textScale;
  return select(d, max(box, d), box > 0.0);
}

/// Extruded, edge-rounded letters occupying 0 <= z <= depth.
fn sd_letters(q: vec3f) -> f32 {
  let b = u.bevel;
  let half = 0.5 * u.depth;
  return op_extrude(sd_text2(q.xy) + b, q.z - half, half - b) - b;
}

fn sd_scene(q: vec3f) -> vec2f {
  let letters_d = sd_letters(q);
  let wall_d = q.z;
  return select(vec2f(wall_d, 0.0), vec2f(letters_d, 1.0), letters_d < wall_d);
}

fn scene_normal(q: vec3f) -> vec3f {
  let e = vec2f(0.5, -0.5);
  return normalize(
    e.xyy * sd_scene(q + e.xyy).x + e.yyx * sd_scene(q + e.yyx).x +
    e.yxy * sd_scene(q + e.yxy).x + e.xxx * sd_scene(q + e.xxx).x);
}

/// Soft shadow toward the light (Quilez, with the "improved" triangulation).
fn soft_shadow(q: vec3f, l: vec3f, max_t: f32) -> f32 {
  var res = 1.0;
  var t = 1.5;
  var ph = 1e10;
  for (var i = 0; i < 40; i++) {
    let p = q + l * t;
    if (p.z > u.depth + 1.0) { break; } // above every letter: nothing more can occlude
    let h = sd_letters(p);
    let y = h * h / (2.0 * ph);
    let d = sqrt(max(h * h - y * y, 0.0));
    res = min(res, d / (u.lightRadius * max(0.0, t - y)));
    ph = h;
    t += clamp(h, 0.75, 40.0);
    if (res < 0.001 || t > max_t) { break; }
  }
  return saturate(res);
}

fn ambient_occlusion(q: vec3f, n: vec3f) -> f32 {
  var occ = 0.0;
  var w = 1.0;
  for (var i = 1; i <= 5; i++) {
    let h = f32(i) * 0.25 * u.depth / 5.0 + 1.0;
    occ += (h - sd_scene(q + n * h).x) * w;
    w *= 0.6;
  }
  return saturate(1.0 - occ / (0.5 * u.depth + 2.0));
}

fn hue_rgb(h: f32) -> vec3f {
  return saturate3(abs(fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
}

fn letter_material(x: f32) -> Material {
  let index = letter_at(x);
  let l = letters[index];
  switch u.materialMode {
    // Alternating metals.
    case 0u: {
      let k = index % 4u;
      var albedo = vec3f(1.0, 0.78, 0.36);            // gold
      if (k == 1u) { albedo = vec3f(0.95, 0.95, 0.97); } // chrome
      if (k == 2u) { albedo = vec3f(0.96, 0.6, 0.45); }  // copper
      if (k == 3u) { albedo = vec3f(0.98, 0.75, 0.7); }  // rose gold
      return Material(albedo, mix(0.18, 0.32, f32(k % 2u)), 1.0);
    }
    // Glossy plastic, one hue per letter.
    case 1u: { return Material(srgb_to_linear(mix(vec3f(0.95), hue_rgb(l.w), 0.75)), 0.28, 0.0); }
    // Single brushed gold.
    default: { return Material(vec3f(1.0, 0.8, 0.42), 0.3, 1.0); }
  }
}

@fragment
fn fs_main(v: FullscreenOut) -> @location(0) vec4f {
  let p = v.position.xy;
  // Ray from above the letters, following the oblique projection.
  let dir = normalize(vec3f(u.shift, -1.0));
  let start_z = u.depth + 2.0;
  // p is where the front face (z = depth) lands; step back up the ray to z = start_z.
  let q = vec3f(p - u.shift * (start_z - u.depth), start_z);
  var hit = vec2f(0.0);
  var t = 0.0;
  for (var i = 0; i < 96; i++) {
    hit = sd_scene(q + dir * t);
    if (hit.x < 0.2) { break; }
    t += hit.x;
    if (t > 4.0 * u.depth + 400.0) { break; }
  }
  let pos = q + dir * t;
  let n = scene_normal(pos);
  let view = -dir;

  var m: Material;
  if (hit.y > 0.5) {
    m = letter_material(pos.x);
  } else {
    // Wall: warm paper with a little tooth.
    let grain = fbm(pos.xy / 9.0, 3) * 0.04;
    m = Material(u.wallColor * (1.0 + grain), 0.85, 0.0);
  }

  let lv = u.light - pos;
  let dist = length(lv);
  let l = lv / dist;
  let shadow = soft_shadow(pos + n * 0.5, l, dist);
  let ao = ambient_occlusion(pos, n);
  let falloff = u.lightColor / (1.0 + (dist * dist) / (u.lightRange * u.lightRange));
  var col = brdf_direct(m, n, view, l) * falloff * shadow * PI;
  col += ibl_shade(m, n, view) * u.envIntensity * ao;

  switch u.debugView {
    case 1u: { col = srgb_to_linear(n * 0.5 + 0.5); }
    case 2u: { col = vec3f(shadow * ao); }
    default: {}
  }
  return vec4f(col, 1.0);
}
