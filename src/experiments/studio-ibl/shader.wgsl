// ---------------------------------------------------------------------------
// Studio Lighting — a product shot lit purely by an environment.
//
//   model:   glTF metal/roughness surface shaded with IBL (SH9 diffuse +
//            prefiltered specular + split-sum LUT + multi-scatter), plus emissive.
//   sky:     the environment itself (optionally blurred), or nothing ("page colour").
//   ground:  an invisible shadow catcher. It only *darkens* what is behind it:
//            - a soft shadow from the environment's dominant light direction
//              (estimated from the SH band-1 vector), via a Poisson-filtered
//              orthographic shadow map;
//            - contact occlusion from a depth map rendered looking *up* from
//              below the floor: each texel stores how high the model's
//              underside is, which maps directly to "how occluded is this
//              bit of floor".
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, VertexIn, brdf.wgsl,
// `Frame`, `IblParams`, ibl.wgsl, gltfWgsl.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> nodes: array<NodeTransform>;
@group(0) @binding(2) var key_map: texture_depth_2d;
@group(0) @binding(3) var contact_map: texture_depth_2d;
@group(0) @binding(4) var shadow_cmp: sampler_comparison;

const POISSON = array<vec2f, 16>(
  vec2f(-0.94201624, -0.39906216), vec2f(0.94558609, -0.76890725),
  vec2f(-0.09418410, -0.92938870), vec2f(0.34495938, 0.29387760),
  vec2f(-0.91588581, 0.45771432), vec2f(-0.81544232, -0.87912464),
  vec2f(-0.38277543, 0.27676845), vec2f(0.97484398, 0.75648379),
  vec2f(0.44323325, -0.97511554), vec2f(0.53742981, -0.47373420),
  vec2f(-0.26496911, -0.41893023), vec2f(0.79197514, 0.19090188),
  vec2f(-0.24188840, 0.99706507), vec2f(-0.81409955, 0.91437590),
  vec2f(0.19984126, 0.78641367), vec2f(0.14383161, -0.14100790),
);

fn model_matrix(id: u32) -> mat4x4f { return frame.modelRoot * nodes[id].model; }

// --- Shadow passes (depth only) -----------------------------------------------

@vertex
fn vs_key(v: GltfVertex, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return frame.keyViewProj * model_matrix(id) * vec4f(v.position, 1.0);
}

@vertex
fn vs_contact(v: GltfVertex, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return frame.contactViewProj * model_matrix(id) * vec4f(v.position, 1.0);
}

// --- Model -------------------------------------------------------------------------

struct ModelOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) tangent: vec4f,
};

@vertex
fn vs_model(v: GltfVertex, @builtin(instance_index) id: u32) -> ModelOut {
  let m = model_matrix(id);
  let wp = m * vec4f(v.position, 1.0);
  // modelRoot is uniform scale + translation, so it commutes with the node's normal matrix.
  let n = (frame.modelRoot * nodes[id].normal * vec4f(v.normal, 0.0)).xyz;
  let t = (m * vec4f(v.tangent.xyz, 0.0)).xyz;
  var out: ModelOut;
  out.position = frame.viewProj * wp;
  out.world_pos = wp.xyz;
  out.normal = n;
  out.uv = v.uv;
  out.tangent = vec4f(t, v.tangent.w);
  return out;
}

@fragment
fn fs_model(v: ModelOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let s = gltf_surface(v.uv, v.world_pos, v.normal, v.tangent, front);
  if (s.alpha < gltf_material.alphaCutoff) { discard; }
  let vdir = normalize(frame.cameraPos - v.world_pos);
  let col = ibl_shade(s.material, s.normal, vdir) * s.occlusion + s.emissive * frame.emissiveStrength;
  return vec4f(col, 1.0);
}

// --- Environment background -------------------------------------------------------

@fragment
fn fs_sky(v: FullscreenOut) -> @location(0) vec4f {
  let ndc = vec2f(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0);
  let far = frame.invViewProj * vec4f(ndc, 1.0, 1.0);
  let dir = normalize(far.xyz / far.w - frame.cameraPos);
  return vec4f(ibl_background(dir, frame.backgroundBlur), 1.0);
}

// --- Ground shadow catcher -----------------------------------------------------------

struct GroundOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
};

@vertex
fn vs_ground(v: VertexIn) -> GroundOut {
  let wp = vec3f(v.position.x * frame.groundSize, 0.0, v.position.z * frame.groundSize);
  var out: GroundOut;
  out.position = frame.viewProj * vec4f(wp, 1.0);
  out.world_pos = wp;
  return out;
}

fn to_shadow_uv(m: mat4x4f, p: vec3f) -> vec3f {
  let c = m * vec4f(p, 1.0);
  let ndc = c.xyz / c.w;
  return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

@fragment
fn fs_ground(g: GroundOut) -> @location(0) vec4f {
  let angle = ign(g.position.xy) * TAU;
  let rot = mat2x2f(cos(angle), sin(angle), -sin(angle), cos(angle));
  var disk = POISSON;

  // Key-light shadow: wide Poisson PCF = a big, soft studio shadow.
  let lp = to_shadow_uv(frame.keyViewProj, g.world_pos);
  var lit = 0.0;
  let key_radius = frame.shadowSoftness / (2.0 * frame.keyExtent);
  for (var i = 0; i < 16; i++) {
    lit += textureSampleCompareLevel(key_map, shadow_cmp, lp.xy + rot * disk[i] * key_radius, lp.z - 0.002);
  }
  lit /= 16.0;

  // Contact occlusion: how close is the model's underside above this point?
  let cp = to_shadow_uv(frame.contactViewProj, g.world_pos);
  let dims = vec2f(textureDimensions(contact_map));
  var occ = 0.0;
  let contact_radius = frame.contactRadius / (2.0 * frame.contactExtent);
  for (var i = 0; i < 16; i++) {
    let uv = cp.xy + rot * disk[i] * contact_radius;
    let px = clamp(vec2i(uv * dims), vec2i(0), vec2i(dims) - 1);
    let d = textureLoad(contact_map, px, 0);
    let height = d * frame.contactRange;
    occ += select(0.0, exp(-height / max(frame.contactFalloff, 1e-3)), d < 1.0);
  }
  occ /= 16.0;

  let key_dark = (1.0 - lit) * frame.shadowOpacity;
  let contact_dark = occ * frame.contactStrength;
  let darkness = 1.0 - (1.0 - key_dark) * (1.0 - contact_dark);
  let fade = 1.0 - smoothstep(frame.groundSize * 0.25, frame.groundSize * 0.5, length(g.world_pos.xz));
  // Premultiplied black with alpha = darkness: only darkens what's behind.
  return vec4f(0.0, 0.0, 0.0, saturate(darkness * fade));
}
