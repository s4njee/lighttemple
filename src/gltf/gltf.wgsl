// ---------------------------------------------------------------------------
// gltf.wgsl — vertex input, node transforms and metal/roughness material
// sampling for meshes from src/gltf/gltf.ts. Materials live in bind group 2.
// Requires (prepended): common.wgsl, brdf.wgsl, `GltfMaterialParams` (added
// automatically when using the exported `gltfWgsl`).
// ---------------------------------------------------------------------------

struct GltfVertex {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  // xyz = tangent, w = bitangent sign; w == 0 means "no tangents supplied".
  @location(3) tangent: vec4f,
};

struct NodeTransform {
  model: mat4x4f,
  // Inverse-transpose of `model`, for normals under non-uniform scale.
  normal: mat4x4f,
};

@group(2) @binding(0) var<uniform> gltf_material: GltfMaterialParams;
@group(2) @binding(1) var gltf_sampler: sampler;
@group(2) @binding(2) var gltf_base_color: texture_2d<f32>;
@group(2) @binding(3) var gltf_metal_rough: texture_2d<f32>;
@group(2) @binding(4) var gltf_normal: texture_2d<f32>;
@group(2) @binding(5) var gltf_occlusion: texture_2d<f32>;
@group(2) @binding(6) var gltf_emissive: texture_2d<f32>;

struct Surface {
  material: Material,
  /// Shading normal (normal-mapped), world space.
  normal: vec3f,
  emissive: vec3f,
  occlusion: f32,
  alpha: f32,
};

/// Samples every material texture and builds the shading normal.
/// Call from uniform control flow (uses implicit-derivative texture sampling).
fn gltf_surface(uv: vec2f, world_pos: vec3f, geom_normal: vec3f, tangent: vec4f, front_facing: bool) -> Surface {
  let base = textureSample(gltf_base_color, gltf_sampler, uv) * gltf_material.baseColorFactor;
  let mr = textureSample(gltf_metal_rough, gltf_sampler, uv);
  let occ = textureSample(gltf_occlusion, gltf_sampler, uv).r;
  let emi = textureSample(gltf_emissive, gltf_sampler, uv).rgb * gltf_material.emissiveFactor;
  var tn = textureSample(gltf_normal, gltf_sampler, uv).xyz * 2.0 - 1.0;
  tn = vec3f(tn.xy * gltf_material.normalScale, tn.z);

  var n = normalize(geom_normal);
  if (!front_facing) { n = -n; } // double-sided back faces

  // Derivative-based cotangent frame (Schüler 2013) for meshes without tangents.
  let dp1 = dpdx(world_pos);
  let dp2 = dpdy(world_pos);
  let duv1 = dpdx(uv);
  let duv2 = dpdy(uv);
  let dp2perp = cross(dp2, n);
  let dp1perp = cross(n, dp1);
  let dt = dp2perp * duv1.x + dp1perp * duv2.x;
  let db = dp2perp * duv1.y + dp1perp * duv2.y;
  let inv_max = inverseSqrt(max(max(dot(dt, dt), dot(db, db)), 1e-20));

  var t = dt * inv_max;
  var b = db * inv_max;
  if (tangent.w != 0.0) {
    t = normalize(tangent.xyz - n * dot(n, tangent.xyz));
    b = cross(n, t) * tangent.w;
  }
  let shading_n = normalize(t * tn.x + b * tn.y + n * tn.z);

  var s: Surface;
  s.material = Material(base.rgb, saturate(mr.g * gltf_material.roughnessFactor), saturate(mr.b * gltf_material.metallicFactor));
  s.normal = shading_n;
  s.emissive = emi;
  s.occlusion = mix(1.0, occ, gltf_material.occlusionStrength);
  s.alpha = base.a;
  return s;
}
