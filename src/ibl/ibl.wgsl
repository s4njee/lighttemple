// ---------------------------------------------------------------------------
// ibl.wgsl — shading with a baked environment (bind group 1).
//
//   ibl_shade(material, n, v)   diffuse (SH9) + specular (prefiltered cube +
//                               split-sum LUT) with multiple-scattering energy
//                               compensation (Fdez-Agüera 2019), so rough metals
//                               don't lose energy and go dark.
//   ibl_background(dir, blur)   the environment itself, for skyboxes/reflections.
//
// Requires (prepended): common.wgsl, brdf.wgsl, the `IblParams` struct.
// Pipelines must put Ibl.bindGroupLayout at group index 1.
// ---------------------------------------------------------------------------

@group(1) @binding(0) var ibl_env: texture_cube<f32>;
@group(1) @binding(1) var ibl_spec: texture_cube<f32>;
@group(1) @binding(2) var ibl_lut: texture_2d<f32>;
@group(1) @binding(3) var ibl_sampler: sampler;
@group(1) @binding(4) var<storage, read> ibl_sh: array<vec4f, 9>;
@group(1) @binding(5) var<uniform> ibl: IblParams;

/// World direction -> environment direction (yaw rotation of the environment).
fn ibl_rotate(d: vec3f) -> vec3f {
  let c = ibl.rotation.x;
  let s = ibl.rotation.y;
  return vec3f(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}

/// Cosine-convolved irradiance E(n) from 9 SH coefficients.
fn ibl_irradiance(n: vec3f) -> vec3f {
  let d = ibl_rotate(n);
  var e = ibl_sh[0].rgb * 0.282095;
  e += ibl_sh[1].rgb * (0.488603 * d.y) + ibl_sh[2].rgb * (0.488603 * d.z) + ibl_sh[3].rgb * (0.488603 * d.x);
  e += ibl_sh[4].rgb * (1.092548 * d.x * d.y) + ibl_sh[5].rgb * (1.092548 * d.y * d.z);
  e += ibl_sh[6].rgb * (0.315392 * (3.0 * d.z * d.z - 1.0));
  e += ibl_sh[7].rgb * (1.092548 * d.x * d.z) + ibl_sh[8].rgb * (0.546274 * (d.x * d.x - d.y * d.y));
  return max(e, vec3f(0.0)) * ibl.intensity;
}

fn ibl_specular_radiance(r: vec3f, roughness: f32) -> vec3f {
  return textureSampleLevel(ibl_spec, ibl_sampler, ibl_rotate(r), roughness * (ibl.specMips - 1.0)).rgb * ibl.intensity;
}

fn ibl_shade(m: Material, n: vec3f, v: vec3f) -> vec3f {
  let nov = max(dot(n, v), 1e-4);
  let r = reflect(-v, n);
  let dfg = textureSampleLevel(ibl_lut, ibl_sampler, vec2f(nov, m.roughness), 0.0).rg;
  let f0 = material_f0(m);

  // Roughness-dependent Fresnel, single- and multiple-scattering terms.
  let fr = max(vec3f(1.0 - m.roughness), f0) - f0;
  let ks = f0 + fr * pow(1.0 - nov, 5.0);
  let fss_ess = ks * dfg.x + dfg.y;
  let ess = dfg.x + dfg.y;
  let ems = 1.0 - ess;
  let f_avg = f0 + (vec3f(1.0) - f0) / 21.0;
  let fms = fss_ess * f_avg / (vec3f(1.0) - ems * f_avg);
  let diffuse_color = m.albedo * (1.0 - m.metallic);
  let kd = diffuse_color * (vec3f(1.0) - fss_ess - fms * ems);

  let irradiance = ibl_irradiance(n) / PI;
  return fss_ess * ibl_specular_radiance(r, m.roughness) + (fms * ems + kd) * irradiance;
}

/// The environment as seen in direction `d`; `blur` 0..1 walks down its mip chain.
fn ibl_background(d: vec3f, blur: f32) -> vec3f {
  return textureSampleLevel(ibl_env, ibl_sampler, ibl_rotate(d), blur * (ibl.envMips - 1.0)).rgb * ibl.intensity;
}
