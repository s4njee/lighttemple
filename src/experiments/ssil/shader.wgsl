// ---------------------------------------------------------------------------
// Screen-space GI: a sunlit room whose indirect light comes from SSIL.
//
//   vs_shadow                  sun shadow map (depth only)
//   vs/fs_geometry             G-buffer (view normal + roughness, albedo +
//                              metallic, depth) *and* the direct sunlight,
//                              which is also the light SSIL bounces
//   fs_composite               direct + albedo × indirect (+ a dim sky term
//                              through the open front), or the comparison
//                              and debug views
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, VertexIn, PROP_WGSL, the
// `Frame` struct, shadow.wgsl, brdf.wgsl, screen-space.wgsl and gtao.wgsl.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(3) var shadow_cmp: sampler_comparison;
@group(0) @binding(4) var gb_depth: texture_depth_2d;
@group(0) @binding(5) var gb_normal: texture_2d<f32>;
@group(0) @binding(6) var gb_albedo: texture_2d<f32>;
@group(0) @binding(7) var direct_tex: texture_2d<f32>;
// Full-resolution indirect light (rgb) + bitmask visibility (a).
@group(0) @binding(8) var gi_tex: texture_2d<f32>;
// Full-resolution GTAO (bent normal + visibility), comparison view only.
@group(0) @binding(9) var ao_tex: texture_2d<f32>;
// History length of the accumulation, for the debug view.
@group(0) @binding(10) var history_tex: texture_2d<f32>;
// Declared for gtao.wgsl's kernels, which this shader doesn't call (no pipeline binds it).
@group(0) @binding(15) var ss_geometry: texture_2d<f32>;

const VIEW_COMBINED: u32 = 0u;
const VIEW_DIRECT: u32 = 1u;
const VIEW_INDIRECT: u32 = 2u;
const VIEW_GTAO: u32 = 3u;
const VIEW_INCOMING: u32 = 4u;
const VIEW_VISIBILITY: u32 = 5u;
const VIEW_HISTORY: u32 = 6u;

// --- Sun shadow map -----------------------------------------------------------

@vertex
fn vs_shadow(v: VertexIn, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return frame.lightViewProj * objects[id].model * vec4f(v.position, 1.0);
}

// --- G-buffer + direct light --------------------------------------------------

struct GeometryOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) id: u32,
};

@vertex
fn vs_geometry(v: VertexIn, @builtin(instance_index) id: u32) -> GeometryOut {
  let m = objects[id].model;
  let wp = m * vec4f(v.position, 1.0);
  var out: GeometryOut;
  out.position = frame.viewProj * wp;
  out.world_pos = wp.xyz;
  out.normal = (m * vec4f(v.normal, 0.0)).xyz;
  out.id = id;
  return out;
}

struct GBufferOut {
  @location(0) normal: vec4f,
  @location(1) albedo: vec4f,
  @location(2) direct: vec4f,
};

fn sun_visibility(world_pos: vec3f, n: vec3f, ndl: f32, frag: vec2f) -> f32 {
  // Normal-offset bias in shadow texels, more at grazing angles.
  let offset = n * frame.normalOffset * frame.shadowTexelWorld * (1.0 - ndl * 0.5);
  let clip = frame.lightViewProj * vec4f(world_pos + offset, 1.0);
  let lp = vec3f(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5, clip.z - frame.depthBias);
  if (!shadow_in_bounds(lp)) { return 1.0; }
  let angle = ign(frag + frame.noiseOffset) * TAU;
  return shadow_poisson(lp, frame.shadowSoftness, angle);
}

@fragment
fn fs_geometry(v: GeometryOut) -> GBufferOut {
  let obj = objects[v.id];
  let n = normalize(v.normal);
  let l = -frame.lightDir;
  let ndl = saturate(dot(n, l));
  let m = Material(obj.color.rgb, obj.color.a, obj.material.x);
  var direct = vec3f(0.0);
  if (ndl > 0.0) {
    let vis = sun_visibility(v.world_pos, n, ndl, v.position.xy) * smoothstep(0.0, 0.08, ndl);
    direct = frame.lightColor.rgb * vis * brdf_direct(m, n, normalize(frame.cameraPos - v.world_pos), l);
  }
  let nv = normalize((frame.view * vec4f(n, 0.0)).xyz);
  return GBufferOut(vec4f(nv, obj.color.a), vec4f(obj.color.rgb, obj.material.x), vec4f(direct, 1.0));
}

// --- Composite -------------------------------------------------------------------

fn heat(t: f32) -> vec3f {
  let x = saturate(t);
  return saturate3(vec3f(1.5 * x, 1.5 * x - 0.5, 0.6 - 2.0 * abs(x - 0.3) + max(0.0, 3.0 * x - 2.0)));
}

@fragment
fn fs_composite(v: FullscreenOut) -> @location(0) vec4f {
  let px = vec2i(v.position.xy);
  let d = textureLoad(gb_depth, px, 0);
  // Daylight outside (seen through the window and the open front).
  if (d >= 1.0) { return vec4f(select(frame.outsideColor.rgb, vec3f(0.0), frame.debugView >= VIEW_INCOMING), 1.0); }
  let gn = textureLoad(gb_normal, px, 0);
  let ga = textureLoad(gb_albedo, px, 0);
  let direct = textureLoad(direct_tex, px, 0).rgb;
  let gi = textureLoad(gi_tex, px, 0);
  let n = normalize((frame.invView * vec4f(gn.xyz, 0.0)).xyz);
  let diffuse = ga.rgb * (1.0 - ga.a);
  let indirect = gi.rgb * frame.giStrength;
  // Skylight through the open front, shadowed by the bitmask visibility.
  let sky = frame.skyColor.rgb * (0.6 + 0.4 * n.z) * gi.a;

  var col = direct + diffuse * (indirect + sky);
  switch frame.debugView {
    case VIEW_DIRECT: { col = direct; }
    case VIEW_INDIRECT: { col = diffuse * indirect; }
    case VIEW_GTAO: {
      // The usual fallback: a flat ambient term darkened by (multi-bounce) GTAO.
      let ao = textureLoad(ao_tex, px, 0).w;
      col = direct + diffuse * frame.fillColor.rgb * gtao_multibounce(ao, diffuse);
    }
    case VIEW_INCOMING: { col = gi.rgb; }
    case VIEW_VISIBILITY: { col = vec3f(gi.a); }
    case VIEW_HISTORY: {
      let hd = vec2f(textureDimensions(history_tex));
      let h = textureLoad(history_tex, vec2i(v.uv * hd), 0).r;
      col = mix(vec3f(0.95, 0.25, 0.15), vec3f(0.08, 0.1, 0.14), saturate((h - 1.0) / max(frame.maxFrames - 1.0, 1.0)));
    }
    default: {}
  }
  return vec4f(col, 1.0);
}
