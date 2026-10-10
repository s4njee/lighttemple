// ---------------------------------------------------------------------------
// Pointer Light — 2.5D SDF soft shadows.
//
// The scene is a flat floor with extruded 2D shapes (an SDF). Lights hover
// above the floor at a given height. For every pixel we:
//   1. derive a surface height + normal from the SDF (bevelled extrusion),
//   2. march toward each light through the SDF, tracking the tightest
//      clearance-to-distance ratio for a soft penumbra (Quilez-style),
//      while letting the ray rise over shapes so shadows have finite length,
//   3. shade with inverse-square falloff and Lambert/Blinn-Phong.
// Outputs linear HDR into the post chain's scene target.
// Requires (prepended): common.wgsl, fullscreen.wgsl, sdf.wgsl, the `Params` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> u: Params;

@group(0) @binding(1) var text_sdf: texture_2d<f32>;
@group(0) @binding(2) var text_sampler: sampler;

// Headline text from a runtime-generated SDF texture (src/sdf/text-sdf.ts).
// textRect = (centre.xy, half-size.zw) in world units; the texture stores
// distance in texels, textScale converts to world units.
fn sd_text(p: vec2f) -> f32 {
  let rel = p - u.textRect.xy;
  let half = u.textRect.zw;
  let box = sd_box(rel, half);
  let uv = clamp(vec2f(rel.x / half.x, -rel.y / half.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
  let d = textureSampleLevel(text_sdf, text_sampler, uv, 0.0).r * u.textScale;
  // Outside the texture, combine two lower bounds (both safe for marching):
  // the distance to the glyphs' own box (texture rect minus padding), which is
  // tight far away, and the edge sample minus the distance travelled to it.
  let glyph_box = sd_box(rel, max(half - vec2f(u.textPad), vec2f(1e-3)));
  return select(d, max(glyph_box, d - box), box > 0.0);
}

fn map_shapes(p: vec2f) -> f32 {
  let t = u.sceneTime;
  var d = 1e9;
  // Tall card, gently swaying.
  d = min(d, sd_round_box(rot2(0.06 * sin(t * 0.4)) * (p - vec2f(-0.62, 0.02 + 0.02 * sin(t * 0.5))), vec2f(0.30, 0.42), 0.05));
  // Big disc.
  d = min(d, sd_circle(p - vec2f(0.42 + 0.03 * sin(t * 0.3), 0.30), 0.20));
  // Small disc.
  d = min(d, sd_circle(p - vec2f(0.12, -0.42 + 0.04 * sin(t * 0.7 + 1.0)), 0.085));
  // Pill.
  d = min(d, sd_round_box(rot2(-0.5 + 0.1 * sin(t * 0.35)) * (p - vec2f(0.78, -0.36)), vec2f(0.24, 0.07), 0.07));
  // Ring.
  d = min(d, sd_ring(p - vec2f(-0.12, 0.55), 0.11, 0.025));
  return d;
}

// Scene SDF in "world" units: y spans [-1, 1], x spans [-aspect, aspect].
fn map(p: vec2f) -> f32 {
  if (u.sceneMode == 1u) { return sd_text(p); }
  return map_shapes(p);
}

fn map_grad(p: vec2f) -> vec2f {
  let e = vec2f(0.0015, 0.0);
  return vec2f(map(p + e.xy) - map(p - e.xy), map(p + e.yx) - map(p - e.yx)) / (2.0 * e.x);
}

// Extruded height of the surface at p (0 on the floor, shapeHeight on top).
fn surface_height(d: f32) -> f32 {
  return u.shapeHeight * smoothstep(0.0, max(u.bevel, 1e-4), -d);
}

// Soft visibility of a spherical light of radius `radius` at `light` (xy, height z)
// from point `p` at height `z0`. Returns 0 (umbra) .. 1 (fully lit).
fn soft_shadow(p: vec2f, z0: f32, light: vec3f, radius: f32) -> f32 {
  let to_light = light.xy - p;
  let s = length(to_light);
  if (s < 1e-4) { return 1.0; }
  let dir = to_light / s;
  // Angular radius of the light as seen from p (small-angle approx).
  let w = radius / length(vec3f(s, 0.0, light.z - z0));
  var res = 1.0;
  var t = 0.004;
  for (var i = 0u; i < u.steps; i++) {
    if (t >= s) { break; }
    let q = p + dir * t;
    let ray_z = mix(z0, light.z, t / s);
    let d = map(q);
    // Vertical clearance over the top of an extrusion counts as clearance too,
    // which is what makes shadows end instead of extending forever.
    let clearance = max(d, ray_z - u.shapeHeight);
    res = min(res, clearance / (w * t));
    if (res < -1.0) { break; }
    // Once the ray is above every extrusion, clearance/(w·t) can only grow, so
    // nothing further along can darken this pixel: shadows cost only their length.
    if (ray_z > u.shapeHeight) { break; }
    t += clamp(abs(clearance), 0.003, 0.08);
  }
  res = max(res, -1.0);
  // Smooth cubic remap of [-1,1] -> [0,1] (lets the penumbra extend inside the umbra).
  return 0.25 * (1.0 + res) * (1.0 + res) * (2.0 - res);
}

struct Light {
  pos: vec3f,
  radius: f32,
  color: vec3f,
};

fn shade_light(l: Light, p: vec2f, z: f32, n: vec3f, albedo: vec3f, shininess: f32, spec_strength: f32, debug_shadow: ptr<function, f32>) -> vec3f {
  if (max(l.color.r, max(l.color.g, l.color.b)) <= 0.0) { return vec3f(0.0); }
  let lv = l.pos - vec3f(p, z);
  let dist2 = dot(lv, lv);
  let ldir = lv * inverseSqrt(dist2);
  let ndl = saturate(dot(n, ldir));
  if (ndl <= 0.0) { return vec3f(0.0); }
  // Rays from anywhere on a shape (including its bevel) start at the top plane so
  // the shape never shadows itself; N·L already darkens bevels facing away.
  let vis = soft_shadow(p, select(z, u.shapeHeight, z > 0.0), l.pos, l.radius);
  *debug_shadow = min(*debug_shadow, vis);
  // Inverse-square falloff, regularised by the light radius so it never explodes.
  let atten = 1.0 / (dist2 + l.radius * l.radius);
  let h = normalize(ldir + vec3f(0.0, 0.0, 1.0));
  let spec = pow(saturate(dot(n, h)), shininess) * spec_strength * (shininess + 8.0) / (8.0 * PI);
  return l.color * atten * vis * ndl * (albedo / PI + spec);
}

@fragment
fn fs_main(v: FullscreenOut) -> @location(0) vec4f {
  let frag = v.position.xy;
  let p = (frag - 0.5 * u.resolution) / u.resolution.y * vec2f(2.0, -2.0);

  let d = map(p);
  let z = surface_height(d);

  // Normal: flat floor, bevelled sides, flat tops. The normal is continuous at
  // the foot of the bevel (smoothstep has zero slope there), so only the
  // material changes across d = 0; blending it over one pixel anti-aliases the edge.
  let px = 2.0 / u.resolution.y;
  let coverage = smoothstep(0.5 * px, -0.5 * px, d);
  var n = vec3f(0.0, 0.0, 1.0);
  if (d < px) {
    let g = map_grad(p);
    let b = max(u.bevel, 1e-4);
    let x = saturate(-d / b);
    let slope = u.shapeHeight * 6.0 * x * (1.0 - x) / b; // derivative of smoothstep
    n = normalize(vec3f(g * slope, 1.0));
  }
  let albedo = mix(u.floorColor.rgb, u.shapeColor.rgb, coverage);
  let shininess = mix(8.0, 48.0, coverage);
  let spec_strength = mix(0.05, u.shapeColor.a, coverage);

  var shadow_dbg = 1.0;
  let l0 = Light(u.light0.xyz, u.light0.w, u.light0Color.rgb);
  let l1 = Light(u.light1.xyz, u.light1.w, u.light1Color.rgb);
  var col = u.ambient.rgb * albedo;
  col += shade_light(l0, p, z, n, albedo, shininess, spec_strength, &shadow_dbg);
  col += shade_light(l1, p, z, n, albedo, shininess, spec_strength, &shadow_dbg);

  // Contact darkening in the crease where floor meets extrusion (cheap fake AO),
  // continuous across the edge: the floor fades in with distance, the bevel
  // foot with height, both reaching 0.55 exactly at d = 0.
  let floor_ao = smoothstep(0.0, max(1.2 * u.shapeHeight, 1e-3), max(d, 0.0));
  let bevel_ao = smoothstep(0.0, 0.4, z / max(u.shapeHeight, 1e-4));
  col *= mix(0.55, 1.0, select(floor_ao, bevel_ao, d < 0.0));

  // Visible light bulbs: a small, very bright core that the bloom pass spreads out.
  var lights = array<Light, 2>(l0, l1);
  for (var i = 0; i < 2; i++) {
    let r = length(p - lights[i].pos.xy);
    col += lights[i].color * 0.6 * exp(-r * r / (lights[i].radius * lights[i].radius * 0.08 + 1e-5));
  }

  if (u.debugView == 1u) { col = vec3f(shadow_dbg); }
  if (u.debugView == 2u) {
    // Classic SDF iso-line view: orange outside, blue inside.
    let base = select(vec3f(0.9, 0.6, 0.3), vec3f(0.35, 0.6, 0.9), d < 0.0);
    let bands = 0.8 + 0.2 * cos(120.0 * d);
    col = srgb_to_linear(base * (1.0 - exp(-6.0 * abs(d))) * bands);
    col = mix(col, vec3f(1.0), 1.0 - smoothstep(0.0, 0.006, abs(d)));
  }
  if (u.debugView == 3u) { col = srgb_to_linear(n * 0.5 + 0.5); }

  // Linear HDR out; exposure, bloom, tonemapping and grain happen in the shared post chain.
  return vec4f(col, 1.0);
}
