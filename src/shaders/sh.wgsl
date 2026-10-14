// ---------------------------------------------------------------------------
// sh.wgsl — real spherical harmonics up to band 2 (9 coefficients) for diffuse
// irradiance: project radiance samples with `sh9_basis`, fold in the clamped
// cosine (`SH_COSINE_BANDS`, Ramamoorthi & Hanrahan 2001), then evaluate
// irradiance E(n) with `sh9_irradiance`. Same basis and ordering as the IBL
// bake (src/ibl/bake.wgsl): index 0 = band 0, 1..3 = band 1 (y, z, x), 4..8 =
// band 2.
//
// Requires (prepended): common.wgsl (PI).
// ---------------------------------------------------------------------------

/// Basis functions Y_k(d) for a unit direction d.
fn sh9_basis(d: vec3f) -> array<f32, 9> {
  return array<f32, 9>(
    0.282095,
    0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x,
    1.092548 * d.x * d.y, 1.092548 * d.y * d.z, 0.315392 * (3.0 * d.z * d.z - 1.0),
    1.092548 * d.x * d.z, 0.546274 * (d.x * d.x - d.y * d.y));
}

/// Clamped-cosine convolution per coefficient (A_0 = π, A_1 = 2π/3, A_2 = π/4):
/// multiply projected radiance by these to get irradiance coefficients.
const SH_COSINE_BANDS = array<f32, 9>(PI, 2.0 * PI / 3.0, 2.0 * PI / 3.0, 2.0 * PI / 3.0, PI / 4.0, PI / 4.0, PI / 4.0, PI / 4.0, PI / 4.0);

/// Hann window for an order-`order` expansion, faded in by `strength` (0 = none):
/// damps the higher bands, trading sharpness for less ringing (Sloan 2008).
fn sh_window(band: u32, order: u32, strength: f32) -> f32 {
  let w = 0.5 * (1.0 + cos(PI * f32(band) / f32(order + 1u)));
  return mix(1.0, w, strength);
}

/// Irradiance E(n) from irradiance coefficients `c` (cosine already folded in),
/// truncated to `order` (1 = L1, 4 coefficients; 2 = L2, all 9) and windowed.
/// Clamped at zero: a truncated expansion can dip below it opposite a bright source.
fn sh9_irradiance(c: array<vec3f, 9>, n: vec3f, order: u32, window: f32) -> vec3f {
  let y = sh9_basis(n);
  var e = c[0] * y[0];
  let w1 = sh_window(1u, order, window);
  e += (c[1] * y[1] + c[2] * y[2] + c[3] * y[3]) * w1;
  if (order >= 2u) {
    let w2 = sh_window(2u, order, window);
    e += (c[4] * y[4] + c[5] * y[5] + c[6] * y[6] + c[7] * y[7] + c[8] * y[8]) * w2;
  }
  return max(e, vec3f(0.0));
}
