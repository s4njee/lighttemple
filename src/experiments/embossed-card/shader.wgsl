// ---------------------------------------------------------------------------
// Embossed Card — a letterpress / embossed business card under a moving light.
//
// One fullscreen pass, no geometry:
//   1. Cast the camera ray against the card's top plane (relief ceiling).
//      Miss -> the table, with the card's soft shadow and contact occlusion.
//   2. Hit -> parallax occlusion mapping through the baked height field:
//      linear search over depth layers, then a secant refinement.
//   3. Normal from the height field's gradient; material = paper/metal/leather
//      base or foil/ink where the design mask says so.
//   4. Light: the pointer light (GGX, inverse-square) with relief self-shadows
//      (march the height field toward the light), plus IBL for reflections.
// Requires (prepended): common.wgsl, noise.wgsl, sdf.wgsl, brdf.wgsl,
// fullscreen.wgsl, `Frame`, `IblParams`, ibl.wgsl.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var height_tex: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;

fn card_uv(xy: vec2f) -> vec2f {
  return vec2f(xy.x, -xy.y) / (2.0 * frame.cardHalf) + 0.5;
}

fn relief(xy: vec2f) -> vec4f {
  return textureSampleLevel(height_tex, lin, card_uv(xy), 0.0);
}

fn card_base_z() -> f32 { return frame.cardLift; }

/// World z of the relief surface for a height-texture value.
fn surface_z(h: f32) -> f32 { return card_base_z() + h * frame.reliefDepth; }

/// Steep parallax + secant refinement. `p` is where the ray enters the relief
/// ceiling; returns the world-space hit on the height field.
fn parallax(p: vec3f, dir: vec3f) -> vec3f {
  let steps = f32(frame.pomSteps);
  // Ray travels from the ceiling (h = 1) down to the floor (h = 0).
  let travel = dir * (frame.reliefDepth / max(-dir.z, 0.05));
  var prev_t = 0.0;
  var prev_gap = 1.0 - relief(p.xy).r; // ray height above surface, in h units (>= 0 above)
  if (prev_gap <= 0.0) { return p; }
  for (var i = 1.0; i <= steps; i += 1.0) {
    let t = i / steps;
    let q = p + travel * t;
    let ray_h = 1.0 - t;
    let gap = ray_h - relief(q.xy).r;
    if (gap <= 0.0) {
      // Secant between the last sample above and the first below the surface.
      let tt = mix(prev_t, t, prev_gap / max(prev_gap - gap, 1e-5));
      return p + travel * tt;
    }
    prev_t = t;
    prev_gap = gap;
  }
  return p + travel;
}

fn relief_normal(xy: vec2f) -> vec3f {
  let e = 1.0 / vec2f(textureDimensions(height_tex)) * 2.0 * frame.cardHalf; // one texel in card units
  let hx = relief(xy + vec2f(e.x, 0.0)).r - relief(xy - vec2f(e.x, 0.0)).r;
  let hy = relief(xy + vec2f(0.0, e.y)).r - relief(xy - vec2f(0.0, e.y)).r;
  let dzdx = hx * frame.reliefDepth / (2.0 * e.x);
  let dzdy = hy * frame.reliefDepth / (2.0 * e.y);
  return normalize(vec3f(-dzdx, -dzdy, 1.0));
}

/// Soft self-shadowing by the relief: march toward the light until the ray
/// clears the relief ceiling, tracking how far it dips below the surface.
fn relief_shadow(p: vec3f, l: vec3f) -> f32 {
  let ceiling = surface_z(1.0);
  if (l.z <= 0.01) { return 0.0; }
  let len = max(ceiling - p.z, 0.0) / l.z;
  if (len <= 0.0) { return 1.0; }
  var vis = 1.0;
  for (var i = 1.0; i <= 12.0; i += 1.0) {
    let t = len * i / 12.0;
    let q = p + l * t;
    let depth_below = surface_z(relief(q.xy).r) - q.z;
    // Penumbra widens with distance travelled (contact-hardening).
    vis = min(vis, saturate(1.0 - depth_below / (0.02 * frame.reliefDepth + t * frame.lightRadius * 0.15)));
  }
  return vis;
}

struct Shaded { color: vec3f, n: vec3f, vis: f32 };

fn shade_card(hit: vec3f, v: vec3f) -> Shaded {
  let r = relief(hit.xy);
  let n = relief_normal(hit.xy);
  let foil = smoothstep(0.2, 0.8, r.g) * f32(frame.foil != 0u);
  let rough_var = (r.b - 0.5) * 0.6;
  var m = Material(frame.baseAlbedo, saturate(frame.baseRoughness + rough_var), frame.baseMetallic);
  m = Material(
    mix(m.albedo, frame.foilAlbedo, foil),
    mix(m.roughness, saturate(frame.foilRoughness + rough_var * 0.3), foil),
    mix(m.metallic, frame.foilMetallic, foil));

  let lv = frame.lightPos - hit;
  let dist2 = dot(lv, lv);
  let l = lv * inverseSqrt(dist2);
  let vis = relief_shadow(hit + n * 1e-4, l);
  let direct = frame.lightColor * brdf_direct(m, n, v, l) * vis / (dist2 + frame.lightRadius * frame.lightRadius);
  // Paper edge: a hint of darkening right at the card's cut edge.
  let edge = smoothstep(0.0, 0.012, -sd_round_box(hit.xy, frame.cardHalf, frame.cardRadius));
  let col = (direct + ibl_shade(m, n, v) * frame.envIntensity) * mix(0.75, 1.0, edge);
  return Shaded(col, n, vis);
}

fn shade_table(q: vec3f, v: vec3f) -> vec3f {
  // Dark walnut-ish desk: long stretched grain.
  let g = fbm(vec2f(q.x * 1.5, q.y * 14.0) + fbm(q.xy * 2.0, 3) * 0.6, 5);
  let albedo = frame.tableAlbedo * (0.75 + 0.35 * g);
  let m = Material(albedo, 0.55 - 0.15 * g, 0.0);
  let n = vec3f(0.0, 0.0, 1.0);

  // Soft shadow of the (lifted) card: where does the ray toward the light cross
  // the card's underside plane, and is that inside the card?
  let lv = frame.lightPos - q;
  let dist2 = dot(lv, lv);
  let l = lv * inverseSqrt(dist2);
  var vis = 1.0;
  if (frame.lightPos.z > frame.cardLift + 1e-3) {
    let s = q.xy + lv.xy * (frame.cardLift / lv.z);
    let penumbra = frame.lightRadius * frame.cardLift / (frame.lightPos.z - frame.cardLift) + 0.002;
    vis = smoothstep(-penumbra, penumbra, sd_round_box(s, frame.cardHalf, frame.cardRadius));
  }
  // Contact occlusion under and around the card's footprint.
  let footprint = sd_round_box(q.xy, frame.cardHalf, frame.cardRadius);
  let ao = mix(0.35, 1.0, smoothstep(-0.01, frame.cardLift * 6.0 + 0.03, footprint));

  let direct = frame.lightColor * brdf_direct(m, n, v, l) * vis / (dist2 + frame.lightRadius * frame.lightRadius);
  return direct + ibl_shade(m, n, v) * frame.envIntensity * ao;
}

@fragment
fn fs_main(fv: FullscreenOut) -> @location(0) vec4f {
  let ndc = vec2f(fv.uv.x * 2.0 - 1.0, 1.0 - fv.uv.y * 2.0);
  let far = frame.invViewProj * vec4f(ndc, 1.0, 1.0);
  let dir = normalize(far.xyz / far.w - frame.cameraPos);
  let v = -dir;

  // Table (z = 0).
  let t_table = -frame.cameraPos.z / dir.z;
  let q = frame.cameraPos + dir * t_table;
  var col = shade_table(q, v);
  var dbg = vec3f(0.0);

  // Card: enter at the relief ceiling.
  let ceiling = surface_z(1.0);
  let t_top = (ceiling - frame.cameraPos.z) / dir.z;
  let p = frame.cameraPos + dir * t_top;
  let card_d = sd_round_box(p.xy, frame.cardHalf, frame.cardRadius);
  // Anti-alias the card silhouette over ~one pixel.
  let px = t_top * frame.pixelAngle;
  let coverage = smoothstep(px, -px, card_d);
  if (coverage > 0.0) {
    let hit = parallax(p, dir);
    let s = shade_card(hit, v);
    col = mix(col, s.color, coverage);
    let r = relief(hit.xy);
    switch frame.debugView {
      case 1u: { dbg = vec3f(r.r); }
      case 2u: { dbg = srgb_to_linear(s.n * 0.5 + 0.5); }
      case 3u: { dbg = vec3f(s.vis); }
      default: {}
    }
    dbg *= coverage;
  }
  if (frame.debugView != 0u) { col = dbg; }
  return vec4f(col, 1.0);
}
