// ---------------------------------------------------------------------------
// Reflective shadow maps (Dachsbacher & Stamminger 2005): the sun's view also
// stores normal and reflected flux; every receiver gathers one bounce from
// texels around its own light-space position, at reduced resolution, then
// bilateral-upsampled (falling back to a full-resolution gather where the
// low-resolution samples don't match).
// Outputs premultiplied linear HDR into the post chain.
// Requires (prepended): common.wgsl, noise.wgsl, brdf.wgsl, shadow.wgsl,
// sun-room.wgsl, VertexIn, PROP_WGSL and the `Frame` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
// The RSM and the low-resolution camera G-buffer.
@group(1) @binding(0) var rsm_depth: texture_depth_2d;
@group(1) @binding(1) var rsm_flux: texture_2d<f32>;
@group(1) @binding(2) var rsm_normal: texture_2d<f32>;
@group(1) @binding(3) var g_pos: texture_2d<f32>;
@group(1) @binding(4) var g_normal: texture_2d<f32>;
// Main pass only.
@group(2) @binding(0) var shadow_map: texture_depth_2d;
@group(2) @binding(1) var shadow_cmp: sampler_comparison;
@group(2) @binding(2) var low_indirect: texture_2d<f32>;
@group(2) @binding(3) var lin_smp: sampler;
// Denoise passes only: the gathered light being blurred.
@group(2) @binding(4) var denoise_src: texture_2d<f32>;

const VIEW_INDIRECT = 1u;
const VIEW_IRRADIANCE = 2u;
const VIEW_DIRECT = 3u;
const VIEW_LOW_RES = 4u;
const VIEW_FALLBACK = 5u;

const UPSAMPLE_BILINEAR = 0u;

struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) id: u32,
};

fn object_vertex(v: VertexIn, id: u32, view_proj: mat4x4f) -> VsOut {
  let m = objects[id].model;
  let wp = m * vec4f(v.position, 1.0);
  var out: VsOut;
  out.position = view_proj * wp;
  out.world_pos = wp.xyz;
  // Boxes are scaled along their own axes, so m·n still points along the face normal.
  out.normal = normalize((m * vec4f(v.normal, 0.0)).xyz);
  out.id = id;
  return out;
}

fn object_albedo(v: VsOut, n: vec3f) -> vec3f {
  let obj = objects[v.id];
  return room_albedo(u32(obj.material.y + 0.5), obj.color.rgb, v.world_pos, n);
}

// --- Light passes ---------------------------------------------------------------

@vertex
fn vs_shadow(v: VertexIn, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return frame.lightViewProj * objects[id].model * vec4f(v.position, 1.0);
}

@vertex
fn vs_light(v: VertexIn, @builtin(instance_index) id: u32) -> VsOut {
  return object_vertex(v, id, frame.lightViewProj);
}

struct RsmOut {
  @location(0) flux: vec4f,
  @location(1) normal: vec4f,
};

/// The RSM: depth (the attachment), world normal, and reflected flux. For a directional light
/// every texel receives the same flux through its footprint, so flux = albedo × sun irradiance
/// (the texel area cancels out in the gather).
@fragment
fn fs_rsm(v: VsOut) -> RsmOut {
  let n = normalize(v.normal);
  // The sun also sees the roof and the outside of the walls, but they face away from every
  // receiver inside, so the gather's cosine term ignores them.
  return RsmOut(vec4f(object_albedo(v, n) * frame.sunColor.rgb, 1.0), vec4f(n, 1.0));
}

// --- Camera G-buffer (reduced resolution) -----------------------------------------

@vertex
fn vs_main(v: VertexIn, @builtin(instance_index) id: u32) -> VsOut {
  return object_vertex(v, id, frame.viewProj);
}

struct GOut {
  @location(0) pos: vec4f,
  @location(1) normal: vec4f,
};

/// World position (w = 1 where there is a receiver) and normal. Cut faces and the window
/// backdrop get w = 0: they take no bounce light.
@fragment
fn fs_gbuffer(v: VsOut) -> GOut {
  let mat = u32(objects[v.id].material.y + 0.5);
  let receiver = mat != ROOM_MAT_SKY && !room_is_cut(v.world_pos);
  return GOut(vec4f(v.world_pos, select(0.0, 1.0, receiver)), vec4f(normalize(v.normal), 0.0));
}

// --- The gather -------------------------------------------------------------------

/// One bounce of sunlight arriving at x (normal n): `count` RSM texels inside a disc of
/// `frame.radius` metres around x's own light-space position. Radii are r = R·ξ with ξ
/// stratified in (0, 1], so samples crowd toward the centre where texels matter most (the
/// paper's importance distribution); each is weighted by ξ to compensate. Angles follow a
/// golden-angle spiral, rotated per pixel and per frame.
///
/// Each texel p is a small area light of flux Φp = ρp·E·A (A = texel area): it adds
/// Φp·cos θp·cos θx / (π d²) at x. Summed over the disc by Monte Carlo (pdf 1/(2π R r) per unit
/// area), A cancels and E(x) ≈ (2 R² / N) Σ ξi·ρi·E·cos θp·cos θx / d².
fn rsm_gather(x: vec3f, n: vec3f, frag: vec2f, count: u32) -> vec3f {
  let clip = frame.lightViewProj * vec4f(x, 1.0);
  let uv0 = vec2f(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
  let dims = vec2f(textureDimensions(rsm_flux));
  let r_uv = frame.radius / (2.0 * frame.rsmExtent);
  let rot = ign(frag + frame.noiseOffset) * TAU;
  let jitter = ign(frag + vec2f(19.0, 47.0) + frame.noiseOffset);
  var e = vec3f(0.0);
  for (var i = 0u; i < count; i++) {
    let xi = (f32(i) + jitter) / f32(count);
    let a = f32(i) * 2.39996323 + rot;
    let uv = uv0 + r_uv * xi * vec2f(cos(a), sin(a));
    if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) { continue; }
    let px = vec2i(uv * dims);
    let flux = textureLoad(rsm_flux, px, 0).rgb;
    if (all(flux == vec3f(0.0))) { continue; }
    // Rebuild the texel's world position from its depth (orthographic light).
    let tuv = (vec2f(px) + 0.5) / dims;
    let depth = textureLoad(rsm_depth, px, 0);
    let wp = frame.invLightViewProj * vec4f(tuv.x * 2.0 - 1.0, 1.0 - tuv.y * 2.0, depth, 1.0);
    let xp = wp.xyz / wp.w;
    let np = textureLoad(rsm_normal, px, 0).xyz;
    let dv = x - xp;
    // Clamped so texels right next to the receiver (at a corner) don't explode.
    let d2 = max(dot(dv, dv), 0.02);
    let geom = max(0.0, dot(np, dv)) * max(0.0, -dot(n, dv)) / (d2 * d2);
    e += flux * (xi * geom);
  }
  return e * (2.0 * frame.radius * frame.radius / f32(count));
}

/// Low-resolution gather: one per G-buffer texel.
@fragment
fn fs_gather(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let px = vec2i(frag.xy);
  let p = textureLoad(g_pos, px, 0);
  if (p.w < 0.5) { return vec4f(0.0); }
  let n = textureLoad(g_normal, px, 0).xyz;
  return vec4f(rsm_gather(p.xyz, n, frag.xy, frame.samples), 1.0);
}

// --- Denoise: separable cross-bilateral blur of the low-resolution gather ----------------

struct DenoiseOut {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) axis: u32,
};

@vertex
fn vs_denoise(@builtin(vertex_index) i: u32, @builtin(instance_index) axis: u32) -> DenoiseOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return DenoiseOut(vec4f(p * 2.0 - 1.0, 0.0, 1.0), axis);
}

/// Gaussian taps along one axis, each weighted down when its G-buffer normal or plane differs,
/// so light does not blur across creases and silhouettes.
@fragment
fn fs_denoise(v: DenoiseOut) -> @location(0) vec4f {
  let px = vec2i(v.position.xy);
  let p0 = textureLoad(g_pos, px, 0);
  if (p0.w < 0.5) { return vec4f(0.0); }
  let n0 = textureLoad(g_normal, px, 0).xyz;
  let dir = select(vec2i(0, 1), vec2i(1, 0), v.axis == 0u);
  let r = i32(frame.denoise);
  let last = vec2i(frame.lowSize) - 1;
  let tol = 0.02 + 0.005 * distance(p0.xyz, frame.cameraPos);
  let inv_2s2 = 1.0 / (2.0 * max(f32(r) * 0.5, 0.5) * max(f32(r) * 0.5, 0.5));
  var sum = vec3f(0.0);
  var wsum = 0.0;
  for (var k = -r; k <= r; k++) {
    let q = clamp(px + dir * k, vec2i(0), last);
    let p = textureLoad(g_pos, q, 0);
    if (p.w < 0.5) { continue; }
    let n = textureLoad(g_normal, q, 0).xyz;
    let w = exp(-f32(k * k) * inv_2s2) * pow(max(dot(n, n0), 0.0), 16.0) * exp(-abs(dot(p.xyz - p0.xyz, n0)) / tol);
    sum += textureLoad(denoise_src, q, 0).rgb * w;
    wsum += w;
  }
  return vec4f(sum / max(wsum, 1e-6), 1.0);
}

// --- Main pass ------------------------------------------------------------------------

struct Upsampled {
  e: vec3f,
  fallback: bool,
};

/// Bilateral upsampling (the paper's trick): average the four low-resolution gathers around
/// this pixel, keeping those whose position and normal match the receiver. With fewer than
/// three matches (silhouettes, thin objects) gather again here, at full resolution.
fn upsample_indirect(x: vec3f, n: vec3f, frag: vec2f) -> Upsampled {
  if (frame.upsample == UPSAMPLE_BILINEAR) {
    return Upsampled(textureSampleLevel(low_indirect, lin_smp, frag / frame.fullSize, 0.0).rgb, false);
  }
  let f = frag * frame.lowSize / frame.fullSize - 0.5;
  let base = floor(f);
  let t = f - base;
  // Matching tolerances grow with distance, like the G-buffer's own precision.
  let tol = 0.03 + 0.01 * distance(x, frame.cameraPos);
  var sum = vec3f(0.0);
  var wsum = 0.0;
  var matches = 0u;
  for (var i = 0u; i < 4u; i++) {
    let o = vec2f(f32(i & 1u), f32(i >> 1u));
    let px = vec2i(clamp(base + o, vec2f(0.0), frame.lowSize - 1.0));
    let p = textureLoad(g_pos, px, 0);
    let np = textureLoad(g_normal, px, 0).xyz;
    let ok = p.w > 0.5 && dot(np, n) > 0.9 && abs(dot(p.xyz - x, n)) < tol && distance(p.xyz, x) < 4.0 * tol;
    if (ok) {
      let w = mix(1.0 - t.x, t.x, o.x) * mix(1.0 - t.y, t.y, o.y) + 1e-3;
      sum += textureLoad(low_indirect, px, 0).rgb * w;
      wsum += w;
      matches++;
    }
  }
  if (matches >= 3u) { return Upsampled(sum / wsum, false); }
  // Half the samples: these pixels are scattered along edges, where TAA and the eye forgive noise.
  return Upsampled(rsm_gather(x, n, frag, max(frame.samples / 2u, 8u)), true);
}

fn sun_visibility(p: vec3f, n: vec3f, ndl: f32) -> f32 {
  let offset = n * frame.normalOffset * frame.shadowTexelWorld * (1.0 - ndl * 0.5);
  let clip = frame.lightViewProj * vec4f(p + offset, 1.0);
  let lp = vec3f(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5, clip.z - 0.0005);
  if (!shadow_in_bounds(lp)) { return 1.0; }
  let texel = 1.0 / f32(textureDimensions(shadow_map).x);
  return shadow_pcf_grid(lp, texel, 1) * smoothstep(0.0, 0.08, ndl);
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4f {
  let obj = objects[v.id];
  let mat = u32(obj.material.y + 0.5);
  let vdir = normalize(frame.cameraPos - v.world_pos);
  if (mat == ROOM_MAT_SKY) {
    return vec4f(room_outdoor(-vdir, frame.skyZenith.rgb, frame.skyHorizon.rgb, frame.outdoorGround.rgb), 1.0);
  }
  let n = normalize(v.normal);
  if (room_is_cut(v.world_pos)) { return vec4f(ROOM_CUT_COLOR * (0.7 + 0.3 * n.y), 1.0); }
  let m = Material(object_albedo(v, n), obj.color.a, obj.material.x);

  let l = frame.toSun;
  let ndl = saturate(dot(n, l));
  var direct = vec3f(0.0);
  if (ndl > 0.0) { direct = frame.sunColor.rgb * sun_visibility(v.world_pos, n, ndl) * brdf_direct(m, n, vdir, l); }

  var up: Upsampled;
  if (frame.debugView == VIEW_LOW_RES) {
    let px = vec2i(v.position.xy * frame.lowSize / frame.fullSize);
    up = Upsampled(textureLoad(low_indirect, px, 0).rgb, false);
  } else {
    up = upsample_indirect(v.world_pos, n, v.position.xy);
  }
  let e = up.e * frame.intensity;
  // A flat fill stands in for the sky light RSM leaves out (it only knows the sun).
  let fill = frame.fillColor.rgb;
  let indirect = (e / PI + fill) * m.albedo * (1.0 - m.metallic);

  var col = direct + indirect;
  switch frame.debugView {
    case VIEW_INDIRECT, VIEW_LOW_RES: { col = e / PI * m.albedo * (1.0 - m.metallic); }
    case VIEW_IRRADIANCE: { col = e / PI; }
    case VIEW_DIRECT: { col = direct; }
    case VIEW_FALLBACK: {
      let shade = 0.25 + 0.5 * luminance(direct + indirect);
      col = select(vec3f(shade), vec3f(1.0, 0.1, 0.6) * (0.4 + shade), up.fallback);
    }
    default: {}
  }
  return vec4f(col, 1.0);
}

// --- Debug overlay: the RSM's three buffers side by side ------------------------------

struct OverlayOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) panel: u32,
};

/// Fullscreen triangle in the current viewport; the instance picks the buffer.
@vertex
fn vs_overlay(@builtin(vertex_index) i: u32, @builtin(instance_index) panel: u32) -> OverlayOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var out: OverlayOut;
  out.position = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(p.x, 1.0 - p.y);
  out.panel = panel;
  return out;
}

@fragment
fn fs_overlay(v: OverlayOut) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(rsm_flux));
  let px = vec2i(clamp(v.uv * dims, vec2f(0.0), dims - 1.0));
  switch v.panel {
    case 0u: {
      // Flux, scaled by the sun so it reads as albedo; black where it is masked out.
      let s = max(max(frame.sunColor.r, frame.sunColor.g), max(frame.sunColor.b, 1e-4));
      return vec4f(textureLoad(rsm_flux, px, 0).rgb / s, 1.0);
    }
    case 1u: { return vec4f(textureLoad(rsm_normal, px, 0).xyz * 0.5 + 0.5, 1.0); }
    default: {
      // World position, normalised to the room's bounding box.
      let tuv = (vec2f(px) + 0.5) / dims;
      let depth = textureLoad(rsm_depth, px, 0);
      let wp = frame.invLightViewProj * vec4f(tuv.x * 2.0 - 1.0, 1.0 - tuv.y * 2.0, depth, 1.0);
      let q = (wp.xyz / wp.w - ROOM_MIN + 0.2) / (ROOM_MAX - ROOM_MIN + 0.4);
      return vec4f(select(saturate3(q), vec3f(0.0), depth >= 1.0), 1.0);
    }
  }
}
