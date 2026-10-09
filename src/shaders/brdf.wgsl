// ---------------------------------------------------------------------------
// brdf.wgsl — metal/roughness PBR (the glTF / Filament model).
//
//   specular: GGX distribution, height-correlated Smith visibility, Schlick Fresnel
//   diffuse:  Lambert, weighted by (1 - F) * (1 - metallic)
//   ambient:  analytic split-sum approximation (Karis 2014), so a single colour
//             or hemisphere "environment" gives plausible, energy-aware reflections
//             without any prefiltered cubemaps (those arrive with E1-S5 IBL).
//
// Requires (prepended): common.wgsl (PI, saturate).
// ---------------------------------------------------------------------------

struct Material {
  albedo: vec3f,
  /// Perceptual roughness 0..1 (squared internally).
  roughness: f32,
  metallic: f32,
};

/// Reflectance at normal incidence: 4% for dielectrics, albedo for metals.
fn material_f0(m: Material) -> vec3f {
  return mix(vec3f(0.04), m.albedo, m.metallic);
}

fn d_ggx(noh: f32, a: f32) -> f32 {
  let a2 = a * a;
  let f = (noh * a2 - noh) * noh + 1.0;
  return a2 / (PI * f * f);
}

fn v_smith_ggx_correlated(nov: f32, nol: f32, a: f32) -> f32 {
  let a2 = a * a;
  let gv = nol * sqrt(nov * nov * (1.0 - a2) + a2);
  let gl = nov * sqrt(nol * nol * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-5);
}

fn f_schlick(f0: vec3f, voh: f32) -> vec3f {
  let f = pow(1.0 - voh, 5.0);
  return f0 + (vec3f(1.0) - f0) * f;
}

/// Outgoing radiance from one light, *excluding* the light's radiance:
/// multiply by (light colour × attenuation × shadow).
fn brdf_direct(m: Material, n: vec3f, v: vec3f, l: vec3f) -> vec3f {
  let nol = saturate(dot(n, l));
  if (nol <= 0.0) { return vec3f(0.0); }
  let h = normalize(v + l);
  let nov = max(dot(n, v), 1e-4);
  let noh = saturate(dot(n, h));
  let voh = saturate(dot(v, h));
  // Clamp roughness: a perfectly smooth GGX lobe aliases into single-pixel sparkles.
  let a = max(m.roughness * m.roughness, 0.002);

  let f0 = material_f0(m);
  let f = f_schlick(f0, voh);
  let spec = d_ggx(noh, a) * v_smith_ggx_correlated(nov, nol, a) * f;
  let kd = (vec3f(1.0) - f) * (1.0 - m.metallic);
  let diffuse = kd * m.albedo / PI;
  return (diffuse + spec) * nol;
}

/// Karis' analytic fit to the split-sum environment BRDF (scale, bias).
fn env_brdf_approx(nov: f32, roughness: f32) -> vec2f {
  let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
  let r = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
  return vec2f(-1.04, 1.04) * a004 + r.zw;
}

/// Ambient term from a diffuse irradiance and a (blurry) specular radiance
/// estimate, e.g. both taken from a sky/ground hemisphere.
fn brdf_ambient(m: Material, n: vec3f, v: vec3f, irradiance: vec3f, specular_radiance: vec3f) -> vec3f {
  let nov = max(dot(n, v), 1e-4);
  let ab = env_brdf_approx(nov, m.roughness);
  let f0 = material_f0(m);
  let spec = specular_radiance * (f0 * ab.x + ab.y);
  let diffuse = irradiance * m.albedo * (1.0 - m.metallic);
  return diffuse + spec;
}
