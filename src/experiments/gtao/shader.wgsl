// ---------------------------------------------------------------------------
// Ground-truth AO: a deferred scene lit only by an environment (IBL), where
// ambient occlusion and bent normals are all the shadowing there is.
//
//   vs/fs_geometry   G-buffer: view normal + roughness, albedo + metallic, depth
//   fs_composite     fullscreen: IBL diffuse through the bent normal, darkened
//                    by (multi-bounce) AO; specular darkened by specular
//                    occlusion; debug views.
// AO itself is computed between the two by the shared screen-space passes.
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, VertexIn, PROP_WGSL, the
// `Frame` struct, brdf.wgsl, the `IblParams` struct, ibl.wgsl,
// screen-space.wgsl and gtao.wgsl.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var gb_depth: texture_depth_2d;
@group(0) @binding(3) var gb_normal: texture_2d<f32>;
@group(0) @binding(4) var gb_albedo: texture_2d<f32>;
// Full-resolution AO: view-space bent normal xyz, visibility w.
@group(0) @binding(5) var ao_tex: texture_2d<f32>;
// History length of the AO accumulation (AO resolution), for the debug view.
@group(0) @binding(6) var history_tex: texture_2d<f32>;
// Declared for gtao.wgsl's kernels, which this shader doesn't call (no pipeline binds it).
@group(0) @binding(15) var ss_geometry: texture_2d<f32>;

const VIEW_LIT: u32 = 0u;
const VIEW_NO_AO: u32 = 1u;
const VIEW_AO: u32 = 2u;
const VIEW_BENT: u32 = 3u;
const VIEW_BENT_ANGLE: u32 = 4u;
const VIEW_NORMALS: u32 = 5u;
const VIEW_HISTORY: u32 = 6u;

// --- G-buffer pass -----------------------------------------------------------

struct GeometryOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) @interpolate(flat) id: u32,
};

@vertex
fn vs_geometry(v: VertexIn, @builtin(instance_index) id: u32) -> GeometryOut {
  let m = objects[id].model;
  var out: GeometryOut;
  out.position = frame.viewProj * m * vec4f(v.position, 1.0);
  // Props use uniform scale, so the upper 3×3 transforms normals correctly.
  out.normal = (m * vec4f(v.normal, 0.0)).xyz;
  out.id = id;
  return out;
}

struct GBufferOut {
  @location(0) normal: vec4f,
  @location(1) albedo: vec4f,
};

@fragment
fn fs_geometry(v: GeometryOut) -> GBufferOut {
  let obj = objects[v.id];
  let n = normalize((frame.view * vec4f(normalize(v.normal), 0.0)).xyz);
  return GBufferOut(vec4f(n, obj.color.a), vec4f(obj.color.rgb, obj.material.x));
}

// --- Composite ---------------------------------------------------------------

fn to_world(d: vec3f) -> vec3f {
  return normalize((frame.invView * vec4f(d, 0.0)).xyz);
}

// ibl_shade() with the diffuse and specular terms occluded separately, and diffuse
// irradiance looked up along `nd` (the bent normal) instead of the shading normal.
fn shade_ambient(m: Material, n: vec3f, nd: vec3f, v: vec3f, diffuse_occ: vec3f, spec_occ: f32) -> vec3f {
  let nov = max(dot(n, v), 1e-4);
  let r = reflect(-v, n);
  let dfg = textureSampleLevel(ibl_lut, ibl_sampler, vec2f(nov, m.roughness), 0.0).rg;
  let f0 = material_f0(m);
  let fr = max(vec3f(1.0 - m.roughness), f0) - f0;
  let ks = f0 + fr * pow(1.0 - nov, 5.0);
  let fss_ess = ks * dfg.x + dfg.y;
  let ess = dfg.x + dfg.y;
  let ems = 1.0 - ess;
  let f_avg = f0 + (vec3f(1.0) - f0) / 21.0;
  let fms = fss_ess * f_avg / (vec3f(1.0) - ems * f_avg);
  let kd = m.albedo * (1.0 - m.metallic) * (vec3f(1.0) - fss_ess - fms * ems);
  let irradiance = ibl_irradiance(nd) / PI;
  return fss_ess * ibl_specular_radiance(r, m.roughness) * spec_occ + (fms * ems + kd) * irradiance * diffuse_occ;
}

fn heat(t: f32) -> vec3f {
  let x = saturate(t);
  return saturate3(vec3f(1.5 * x, 1.5 * x - 0.5, 0.6 - 2.0 * abs(x - 0.3) + max(0.0, 3.0 * x - 2.0)));
}

@fragment
fn fs_composite(v: FullscreenOut) -> @location(0) vec4f {
  let px = vec2i(v.position.xy);
  let d = textureLoad(gb_depth, px, 0);
  if (d >= 1.0) {
    if (frame.backdrop == 1u && (frame.debugView == VIEW_LIT || frame.debugView == VIEW_NO_AO)) {
      let dir = to_world(ss_view_pos(v.uv, 1.0, frame.projScale, frame.jitter));
      return vec4f(ibl_background(dir, frame.backdropBlur), 1.0);
    }
    return vec4f(0.0);
  }
  let z = ss_linear_depth(d, frame.near, frame.far);
  let view_pos = ss_view_pos(v.uv, z, frame.projScale, frame.jitter);
  let world = (frame.invView * vec4f(view_pos, 1.0)).xyz;
  let gn = textureLoad(gb_normal, px, 0);
  let ga = textureLoad(gb_albedo, px, 0);
  let n = to_world(gn.xyz);
  let vdir = normalize(frame.cameraPos - world);
  let m = Material(ga.rgb, gn.w, ga.a);

  let ao = textureLoad(ao_tex, px, 0);
  let vis = pow(saturate(ao.w), frame.aoPower);
  let bent = select(n, to_world(ao.xyz), dot(ao.xyz, ao.xyz) > 1e-6);
  var nd = n;
  if (frame.bentNormals != 0u) { nd = normalize(mix(n, bent, frame.bentStrength)); }
  var diffuse_occ = vec3f(vis);
  if (frame.multiBounce != 0u) { diffuse_occ = gtao_multibounce(vis, m.albedo * (1.0 - m.metallic)); }
  var spec_occ = 1.0;
  if (frame.specOcclusion != 0u) { spec_occ = specular_occlusion(max(dot(n, vdir), 0.0), vis, m.roughness); }
  if (frame.debugView == VIEW_NO_AO) {
    nd = n;
    diffuse_occ = vec3f(1.0);
    spec_occ = 1.0;
  }
  var col = shade_ambient(m, n, nd, vdir, diffuse_occ, spec_occ);

  switch frame.debugView {
    case VIEW_AO: { col = vec3f(vis); }
    case VIEW_BENT: { col = bent * 0.5 + 0.5; }
    case VIEW_BENT_ANGLE: {
      // How far the bent normal leans away from the surface normal (0° black .. 60°+ white-hot).
      col = heat(acos(clamp(dot(bent, n), -1.0, 1.0)) / (PI / 3.0));
    }
    case VIEW_NORMALS: { col = n * 0.5 + 0.5; }
    case VIEW_HISTORY: {
      let hd = vec2f(textureDimensions(history_tex));
      let h = textureLoad(history_tex, vec2i(v.uv * hd), 0).r;
      // Fresh history (just disoccluded) red, converged history dark.
      col = mix(vec3f(0.95, 0.25, 0.15), vec3f(0.08, 0.1, 0.14), saturate((h - 1.0) / max(frame.maxFrames - 1.0, 1.0)));
    }
    default: {}
  }

  // The floor fades out radially: into the environment, or (as coverage) into the page colour,
  // which the post chain composites after tonemapping so the two match exactly.
  let lit = frame.debugView == VIEW_LIT || frame.debugView == VIEW_NO_AO;
  let fade = select(0.0, smoothstep(frame.fadeStart, frame.fadeEnd, length(world.xz)), lit);
  if (frame.backdrop == 1u && lit) {
    return vec4f(mix(col, ibl_background(normalize(world - frame.cameraPos), frame.backdropBlur), fade), 1.0);
  }
  let coverage = 1.0 - fade;
  return vec4f(col * coverage, coverage);
}
