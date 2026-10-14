// ---------------------------------------------------------------------------
// Blob & Baked Shadows — the cheap end of the shadow spectrum, compared with a
// real-time shadow map on the same scene.
//
//   mode 0  blob:     every object gets a soft ellipse decal on the ground,
//                     offset and stretched along the light, growing and fading
//                     with height. No shadow passes at all.
//   mode 1  baked:    static objects' sun shadow and sky occlusion were baked
//                     once into a ground lightmap (many shadow-map lookups over
//                     jittered light directions); moving objects still get blobs.
//   mode 2  real-time: one shadow map per frame, Poisson-filtered.
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, VertexIn, Object (props),
// brdf.wgsl, shadow.wgsl, the `Frame` and `Bake` structs.
// ---------------------------------------------------------------------------

struct Blob {
  // xy = centre on the ground (world xz), zw = unit axis along the light
  center_axis: vec4f,
  // x, y = semi-axes along / across the light, z = opacity, w = edge softness (0..1)
  shape: vec4f,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(3) var shadow_cmp: sampler_comparison;
@group(0) @binding(4) var lightmap: texture_2d<f32>;
@group(0) @binding(5) var lin: sampler;
@group(0) @binding(6) var<storage, read> blobs: array<Blob>;

// --- Real-time shadow pass -------------------------------------------------------

@vertex
fn vs_shadow(v: VertexIn, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return frame.lightViewProj * objects[id].model * vec4f(v.position, 1.0);
}

fn to_shadow_uv(m: mat4x4f, p: vec3f) -> vec3f {
  let c = m * vec4f(p, 1.0);
  let ndc = c.xyz / c.w;
  return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

fn realtime_shadow(p: vec3f, n: vec3f, ndl: f32, frag: vec2f) -> f32 {
  let offset = n * frame.shadowTexelWorld * 1.5 * (1.0 - 0.5 * ndl);
  let lp = to_shadow_uv(frame.lightViewProj, p + offset) - vec3f(0.0, 0.0, 0.0005);
  if (!shadow_in_bounds(lp)) { return 1.0; }
  let texel = 1.0 / f32(textureDimensions(shadow_map).x);
  return shadow_poisson(lp, 2.0 * texel, ign(frag) * TAU);
}

// --- Lit objects -------------------------------------------------------------------

struct ObjOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) id: u32,
};

@vertex
fn vs_object(v: VertexIn, @builtin(instance_index) id: u32) -> ObjOut {
  let m = objects[id].model;
  let wp = m * vec4f(v.position, 1.0);
  var out: ObjOut;
  out.position = frame.viewProj * wp;
  out.world_pos = wp.xyz;
  // Every prop is uniformly scaled (boxes are built at size), so the upper 3×3 works for normals.
  out.normal = (m * vec4f(v.normal, 0.0)).xyz;
  out.id = id;
  return out;
}

fn hemisphere(n: vec3f, ao: f32) -> vec3f {
  return mix(frame.groundBounce, frame.skyColor, n.y * 0.5 + 0.5) * ao;
}

@fragment
fn fs_object(v: ObjOut) -> @location(0) vec4f {
  let obj = objects[v.id];
  let n = normalize(v.normal);
  let l = frame.toLight;
  let vdir = normalize(frame.cameraPos - v.world_pos);
  let ndl = saturate(dot(n, l));
  // Only the real-time mode can shadow objects (from each other and themselves).
  var vis = 1.0;
  if (frame.mode == 2u) { vis = realtime_shadow(v.world_pos, n, ndl, v.position.xy) * smoothstep(0.0, 0.08, ndl); }
  // A cheap ambient darkening toward the ground stands in for AO in every mode.
  let ground_ao = mix(0.55, 1.0, saturate(v.world_pos.y * 1.6 + 0.25));
  let m = Material(obj.color.rgb, obj.color.a, obj.material.x);
  let r = reflect(-vdir, n);
  let env = mix(frame.groundBounce, frame.skyColor, smoothstep(-0.4, 0.4, r.y));
  var col = brdf_ambient(m, n, vdir, hemisphere(n, ground_ao), env * ground_ao) + frame.sunColor * vis * brdf_direct(m, n, vdir, l);
  if (frame.debugView == 1u) { col = vec3f(0.5 + 0.5 * vis) * 0.6; }
  return vec4f(col, 1.0);
}

// --- Ground ------------------------------------------------------------------------

struct GroundOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
};

@vertex
fn vs_ground(v: VertexIn) -> GroundOut {
  var out: GroundOut;
  out.world_pos = v.position;
  out.position = frame.viewProj * vec4f(v.position, 1.0);
  return out;
}

/// Baked lightmap: x = sun visibility, y = sky visibility (both already averaged).
fn baked(p: vec3f) -> vec2f {
  let uv = p.xz / (2.0 * frame.bakeExtent) + 0.5;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return vec2f(1.0); }
  let s = textureSampleLevel(lightmap, lin, uv, 0.0);
  // Progressive bake: z/w hold how many sun/sky samples have landed so far.
  return vec2f(select(1.0, s.x / s.z, s.z > 0.0), select(1.0, s.y / s.w, s.w > 0.0));
}

@fragment
fn fs_ground(g: GroundOut) -> @location(0) vec4f {
  let n = vec3f(0.0, 1.0, 0.0);
  let ndl = saturate(frame.toLight.y);

  var sun = 1.0;
  var sky = 1.0;
  if (frame.mode == 1u) {
    let b = baked(g.world_pos);
    sun = b.x;
    sky = b.y;
  } else if (frame.mode == 2u) {
    sun = realtime_shadow(g.world_pos, n, ndl, g.position.xy);
  }
  let vdir = normalize(frame.cameraPos - g.world_pos);
  let m = Material(frame.groundColor, 0.8, 0.0);
  var col = brdf_ambient(m, n, vdir, frame.skyColor * sky, frame.skyColor * sky) + frame.sunColor * sun * brdf_direct(m, n, vdir, frame.toLight);
  if (frame.debugView == 1u) { col = vec3f(sun * 0.8 + 0.2 * sky); }
  // Radial fade into the page colour (applied after tonemapping by the post chain).
  let fade = select(0.0, smoothstep(frame.fadeStart, frame.fadeEnd, length(g.world_pos.xz)), frame.debugView == 0u);
  let coverage = 1.0 - fade;
  return vec4f(col * coverage, coverage);
}

// --- Blob decals -------------------------------------------------------------------

struct BlobOut {
  @builtin(position) position: vec4f,
  // Position in the blob's frame, in semi-axis units (|local| = 1 on the ellipse).
  @location(0) local: vec2f,
  @location(1) @interpolate(flat) id: u32,
};

@vertex
fn vs_blob(@builtin(vertex_index) vi: u32, @builtin(instance_index) id: u32) -> BlobOut {
  // Two triangles covering [-1, 1]², slightly larger so the soft rim fits.
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let b = blobs[id];
  let q = corners[vi] * 1.05;
  let axis = b.center_axis.zw;
  let side = vec2f(-axis.y, axis.x);
  let xz = b.center_axis.xy + axis * q.x * b.shape.x + side * q.y * b.shape.y;
  var out: BlobOut;
  out.position = frame.viewProj * vec4f(xz.x, 0.004, xz.y, 1.0);
  out.local = q;
  out.id = id;
  return out;
}

@fragment
fn fs_blob(v: BlobOut) -> @location(0) vec4f {
  let b = blobs[v.id];
  let d = length(v.local);
  // Soft ellipse: solid core, smooth falloff over the outer `softness` of the radius, and a little
  // lighter toward the far end (+x points away from the object along the light), like a penumbra.
  let tail = 1.0 - 0.4 * saturate(v.local.x * 0.5 + 0.5);
  let a = b.shape.z * tail * (1.0 - smoothstep(1.0 - b.shape.w, 1.0, d));
  // Premultiplied black: darkens the lit ground it lands on.
  return vec4f(0.0, 0.0, 0.0, a);
}

// --- Bake: many shadow maps accumulated into the ground lightmap ---------------------

@group(1) @binding(0) var<uniform> bake: Bake;

@vertex
fn vs_bake_depth(v: VertexIn, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return bake.viewProj * objects[id].model * vec4f(v.position, 1.0);
}

@group(1) @binding(1) var bake_depth: texture_depth_2d;
@group(1) @binding(2) var bake_cmp: sampler_comparison;

@fragment
fn fs_bake_accumulate(v: FullscreenOut) -> @location(0) vec4f {
  // Lightmap texel -> ground point (y = 0).
  let xz = (v.uv - 0.5) * 2.0 * bake.extent;
  let lp = to_shadow_uv(bake.viewProj, vec3f(xz.x, 0.0, xz.y));
  var vis = 1.0;
  if (all(lp.xy > vec2f(0.0)) && all(lp.xy < vec2f(1.0)) && lp.z < 1.0) {
    vis = textureSampleCompareLevel(bake_depth, bake_cmp, lp.xy, lp.z - 0.001);
  }
  // Additively blended: x/z accumulate sun visibility / sun samples, y/w the same for the sky.
  return vec4f(vis * bake.kind.x, vis * bake.kind.y, bake.kind.x, bake.kind.y);
}

// --- Debug overlay: the lightmap in a corner -------------------------------------------

@fragment
fn fs_overlay(v: FullscreenOut) -> @location(0) vec4f {
  let s = textureSampleLevel(lightmap, lin, v.uv, 0.0);
  let sun = select(1.0, s.x / s.z, s.z > 0.0);
  let sky = select(1.0, s.y / s.w, s.w > 0.0);
  return vec4f(sun, sky, sky, 1.0);
}
