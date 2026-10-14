// ---------------------------------------------------------------------------
// SDF soft shadows (3D): a fullscreen ray marcher whose sun shadows come from
// cone-tracing the distance field (analytic or baked into a 3D texture), with
// a shadow map + PCSS of the very same scene for comparison.
// Outputs premultiplied linear HDR (alpha = coverage) and depth (for TAA).
// Requires (prepended): common.wgsl, fullscreen.wgsl, sdf.wgsl, brdf.wgsl,
// the `Frame` struct, scene.wgsl, shadow.wgsl.
// ---------------------------------------------------------------------------

@group(0) @binding(1) var baked: texture_3d<f32>;
@group(0) @binding(2) var baked_smp: sampler;
@group(0) @binding(3) var shadow_map: texture_depth_2d;
@group(0) @binding(4) var shadow_cmp: sampler_comparison;

const MODE_HARD = 0u;
const MODE_CLASSIC = 1u;
const MODE_IMPROVED = 2u;
const MODE_BAKED = 3u;
const MODE_PCSS = 4u;
const MODE_SPLIT = 5u;

const FAR = 40.0;
/// Height of the distance-field slice debug view.
const SLICE_Y = 0.3;

// --- Baked distance --------------------------------------------------------

/// Static scene distance from the 3D texture (hardware trilinear). Outside the
/// volume, two lower bounds are combined: the distance to the volume's box plus
/// the empty margin the bake leaves around every object, and the value at the
/// nearest point on the box minus the distance to it. The second makes the
/// field continuous across the boundary, so rays leaving the volume don't see
/// a false "near miss" there.
fn baked_static(p: vec3f) -> f32 {
  let hs = 0.5 * frame.boundsSize;
  let uvw = (p - frame.boundsMin) / frame.boundsSize;
  let inner = textureSampleLevel(baked, baked_smp, saturate3(uvw), 0.0).r;
  let outside = sd_box3(p - frame.boundsMin - hs, hs);
  if (outside <= 0.0) { return inner; }
  return max(outside + 0.1, inner - outside);
}

/// Distance used by shadow rays: analytic, or baked + the analytic extras.
fn map_shadow(p: vec3f, use_baked: bool) -> f32 {
  var d = min(p.y, map_dynamic(p));
  if (use_baked) { return min(d, baked_static(p)); }
  return min(d, map_static(p).x);
}

// --- Shadow rays -------------------------------------------------------------
// All return (visibility, steps taken). `w` = tan of the sun's angular radius:
// a ray passing at distance h from an occluder, t along the ray, sees the
// occluder cover the fraction ~ h / (w·t) of the sun's disc.

const T_MIN = 0.015;
const T_MAX = 14.0;
/// Nothing is taller than this, so a sun-bound ray above it can stop: shadow
/// cost then scales with how much scene the ray actually crosses.
const SCENE_TOP = 2.45;

fn escaped(p: vec3f) -> bool { return p.y > SCENE_TOP; }

fn sdf_shadow_hard(ro: vec3f, rd: vec3f) -> vec2f {
  var t = T_MIN;
  var i = 0u;
  for (; i < frame.shadowSteps; i++) {
    let h = map_shadow(ro + rd * t, false);
    if (h < 0.0005) { return vec2f(0.0, f32(i + 1u)); }
    t += h;
    if (t > T_MAX || escaped(ro + rd * t)) { break; }
  }
  return vec2f(1.0, f32(i));
}

/// Quilez 2010: track the smallest h/(w·t) along the ray. It only ever darkens
/// *outside* the hard shadow (outer penumbra), and under-samples the closest
/// approach between steps, which shows as banding on sharp corners.
fn sdf_shadow_classic(ro: vec3f, rd: vec3f) -> vec2f {
  var res = 1.0;
  var t = T_MIN;
  var i = 0u;
  for (; i < frame.shadowSteps; i++) {
    let h = map_shadow(ro + rd * t, false);
    if (h < 0.0005) { return vec2f(0.0, f32(i + 1u)); }
    res = min(res, h / (frame.sunTan * t));
    t += clamp(h, 0.01, 0.4);
    if (t > T_MAX || escaped(ro + rd * t)) { break; }
  }
  let v = saturate(res);
  return vec2f(v * v * (3.0 - 2.0 * v), f32(i));
}

/// The improved estimate (Aaltonen 2018): the previous and current distance
/// spheres intersect in a circle, whose radius is a better guess at how close
/// the ray passed to the surface between the two samples. (Written for any
/// step length `s`; with a pure sphere-tracing step s = ph it reduces to the
/// familiar y = h²/2ph, but our steps are clamped.) Then, after Quilez 2020,
/// the ray keeps marching *into* occluders (negative distances) so the
/// penumbra extends inside the hard edge as well: centred, like a real one.
fn sdf_shadow_improved(ro: vec3f, rd: vec3f, use_baked: bool) -> vec2f {
  var res = 1.0;
  var t = T_MIN;
  var ph = -1.0;
  var s = 0.0;
  var i = 0u;
  for (; i < frame.shadowSteps; i++) {
    let h = map_shadow(ro + rd * t, use_baked);
    var est = h / (frame.sunTan * t);
    if (h > 0.0 && ph > 0.0) {
      let y = (s * s + h * h - ph * ph) / (2.0 * s);
      if (y > 0.0 && y < h) {
        est = sqrt(h * h - y * y) / (frame.sunTan * max(t - y, 1e-4));
      }
    }
    res = min(res, est);
    ph = h;
    s = clamp(h, 0.01, 0.5);
    t += s;
    if (res < -1.0 || t > T_MAX || escaped(ro + rd * t)) { break; }
  }
  res = max(res, -1.0);
  // Smooth cubic remap of [-1, 1] -> [0, 1].
  return vec2f(0.25 * (1.0 + res) * (1.0 + res) * (2.0 - res), f32(i));
}

/// Shadow map + PCSS of the same scene (the map is ray-marched from the sun).
fn shadow_pcss_lookup(p: vec3f, n: vec3f, frag: vec2f) -> f32 {
  let offset = n * frame.normalOffset * frame.shadowTexelWorld;
  let clip = frame.lightViewProj * vec4f(p + offset, 1.0);
  let lp = vec3f(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5, clip.z - frame.depthBias * frame.shadowTexelWorld / frame.shadowDepthRange);
  if (!shadow_in_bounds(lp)) { return 1.0; }
  let texel = 1.0 / f32(textureDimensions(shadow_map).x);
  let angle = ign(frag + frame.noiseOffset) * TAU;
  return shadow_pcss(lp, texel, frame.sunTan, frame.shadowDepthRange, 1.0 / (2.0 * frame.shadowExtent), angle);
}

// --- Shading -----------------------------------------------------------------

fn normal_at(p: vec3f) -> vec3f {
  // Tetrahedral central differences: 4 taps instead of 6.
  let e = vec2f(1.0, -1.0) * 0.0007;
  return normalize(
    e.xyy * map(p + e.xyy).x + e.yyx * map(p + e.yyx).x + e.yxy * map(p + e.yxy).x + e.xxx * map(p + e.xxx).x,
  );
}

fn ambient_occlusion(p: vec3f, n: vec3f) -> f32 {
  var occ = 0.0;
  var w = 1.0;
  for (var i = 1; i <= 5; i++) {
    let h = 0.02 + 0.09 * f32(i);
    occ += (h - map(p + n * h).x) * w;
    w *= 0.7;
  }
  return saturate(1.0 - 1.6 * occ);
}

fn material(id: f32, p: vec3f) -> Material {
  switch u32(id + 0.5) {
    case 1u: { return Material(vec3f(0.62, 0.55, 0.46), 0.85, 0.0); }
    case 2u: { return Material(vec3f(0.95, 0.64, 0.54), 0.3, 1.0); }
    case 3u: { return Material(vec3f(0.62, 0.25, 0.15), 0.65, 0.0); }
    case 4u: { return Material(vec3f(0.62, 0.63, 0.66), 0.35, 1.0); }
    case 5u: { return Material(vec3f(0.17, 0.22, 0.3), 0.55, 0.0); }
    case 6u: { return Material(vec3f(0.85, 0.82, 0.78), 0.2, 0.0); }
    default: {
      // Floor: pale plaster tiles with faint joints.
      let g = abs(fract(p.xz / 1.2) - 0.5);
      let joint = smoothstep(0.494, 0.4985, max(g.x, g.y));
      return Material(vec3f(0.64, 0.6, 0.55) * (1.0 - 0.18 * joint), 0.8, 0.0);
    }
  }
}

/// Turbo-ish ramp for step-count heatmaps.
fn heat(x: f32) -> vec3f {
  let t = saturate(x);
  return saturate3(vec3f(
    1.5 - abs(4.0 * t - 3.0),
    1.5 - abs(4.0 * t - 2.0),
    1.5 - abs(4.0 * t - 1.0),
  )) * (0.25 + 0.75 * step(0.001, t));
}

struct Out {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(v: FullscreenOut) -> Out {
  let ndc = vec2f(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0);
  let far = frame.invViewProj * vec4f(ndc, 1.0, 1.0);
  let ro = frame.cameraPos;
  let rd = normalize(far.xyz / far.w - ro);

  // Primary ray: sphere tracing with a distance-relative hit threshold.
  var t = 0.05;
  var hit = vec2f(-1.0);
  var primary_steps = 0u;
  for (; primary_steps < frame.maxSteps; primary_steps++) {
    let r = map(ro + rd * t);
    if (r.x < 0.0004 * t) {
      hit = vec2f(t, r.y);
      break;
    }
    t += r.x;
    if (t > FAR) { break; }
  }

  var out: Out;
  if (hit.x < 0.0) {
    out.color = vec4f(0.0);
    out.depth = 1.0;
    if (frame.debugView == 3u) { out.color = vec4f(heat(f32(primary_steps) / f32(frame.maxSteps)), 1.0); }
    return out;
  }
  let p = ro + rd * hit.x;
  let n = normal_at(p);
  let clip = frame.viewProj * vec4f(p, 1.0);
  out.depth = clip.z / clip.w;

  // Which shadow technique this pixel uses (split mode: left SDF, right PCSS).
  var mode = frame.shadowMode;
  if (mode == MODE_SPLIT) { mode = select(MODE_PCSS, MODE_IMPROVED, v.uv.x < frame.splitX); }

  let l = frame.sunDir;
  let ndl = dot(n, l);
  var sh = vec2f(1.0, 0.0);
  if (ndl > 0.0 || frame.debugView != 0u) {
    let origin = p + n * 0.004;
    switch mode {
      case 0u: { sh = sdf_shadow_hard(origin, l); }
      case 1u: { sh = sdf_shadow_classic(origin, l); }
      case 2u: { sh = sdf_shadow_improved(origin, l, false); }
      case 3u: { sh = sdf_shadow_improved(origin, l, true); }
      default: { sh = vec2f(shadow_pcss_lookup(p, n, v.position.xy), 0.0); }
    }
  }
  let vis = sh.x * smoothstep(0.0, 0.05, ndl);

  let m = material(hit.y, p);
  let vdir = -rd;
  let ao = ambient_occlusion(p, n);
  let irradiance = mix(frame.groundColor.rgb, frame.skyColor.rgb, n.y * 0.5 + 0.5) * ao;
  let refl = reflect(rd, n);
  let edge = mix(0.05, 0.8, m.roughness);
  let env = mix(frame.groundColor.rgb, frame.skyColor.rgb, smoothstep(-edge, edge, refl.y)) * ao;
  var col = brdf_ambient(m, n, vdir, irradiance, env) + frame.sunColor.rgb * vis * brdf_direct(m, n, vdir, l);

  switch frame.debugView {
    case 1u: { col = vec3f(vis); }
    case 2u: { col = select(heat(sh.y / f32(frame.shadowSteps)), vec3f(0.1), mode == MODE_PCSS); }
    case 3u: { col = heat(f32(primary_steps) / f32(frame.maxSteps)); }
    case 4u: {
      // A horizontal slice through the distance field the shadow rays see
      // (baked in Baked mode), drawn on the floor: iso-lines every 5 cm, the
      // surface itself in white, inside in blue.
      col = vec3f(0.3) * (0.4 + 0.6 * vis);
      if (hit.y == MAT_FLOOR) {
        let d = map_shadow(vec3f(p.x, SLICE_Y, p.z), mode == MODE_BAKED);
        let base = select(vec3f(0.9, 0.6, 0.3), vec3f(0.35, 0.6, 0.9), d < 0.0);
        let bands = 0.75 + 0.25 * smoothstep(0.0, 0.12, abs(fract(d / 0.05) - 0.5));
        col = srgb_to_linear(base * bands * (1.0 - 0.5 * exp(-8.0 * abs(d))));
        col = mix(col, vec3f(1.0), 1.0 - smoothstep(0.0, 0.006, abs(d)));
      }
    }
    default: {}
  }
  // Split view: a hairline divider.
  if (frame.shadowMode == MODE_SPLIT && abs(v.uv.x - frame.splitX) * frame.resolution.x < 1.0) { col = vec3f(4.0); }

  // The floor dissolves into the page colour, matched after tonemapping by the post chain.
  let fade = select(0.0, smoothstep(frame.fadeStart, frame.fadeEnd, length(p.xz)), frame.debugView == 0u && hit.y == MAT_FLOOR);
  let coverage = 1.0 - fade;
  out.color = vec4f(col * coverage, coverage);
  return out;
}

// --- Light-space depth for the PCSS comparison -----------------------------------

/// One ray per shadow-map texel, marched from the sun's near plane into the
/// scene; the hit depth is written exactly as a rasterised depth pass would.
@fragment
fn fs_light_depth(v: FullscreenOut) -> @builtin(frag_depth) f32 {
  let ndc = vec2f(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0);
  let start = frame.invLightViewProj * vec4f(ndc, 0.0, 1.0);
  let ro = start.xyz / start.w;
  let rd = -frame.sunDir;
  var t = 0.0;
  for (var i = 0; i < 160; i++) {
    let d = map(ro + rd * t).x;
    if (d < 0.0005) {
      let clip = frame.lightViewProj * vec4f(ro + rd * t, 1.0);
      return clip.z / clip.w;
    }
    t += d;
    if (t > frame.shadowDepthRange) { break; }
  }
  return 1.0;
}
