// ---------------------------------------------------------------------------
// optics.wgsl — wave-optics approximations for iridescent and holographic surfaces.
//   spectral_rgb(λ)        wavelength (nm) -> linear RGB (Zucconi's 6-bump fit)
//   thin_film(...)         interference colour of a thin dielectric film
//   diffraction_grating()  rainbow from a reflective grating (holographic foil)
// Requires (prepended): common.wgsl.
// ---------------------------------------------------------------------------

fn bump3(x: vec3f, y_offset: vec3f) -> vec3f {
  return saturate3(vec3f(1.0) - x * x - y_offset);
}

/// Visible wavelength in nm (≈400..700) -> linear RGB with roughly equal energy.
fn spectral_rgb(w: f32) -> vec3f {
  let x = saturate((w - 400.0) / 300.0);
  let c1 = vec3f(3.54585104, 2.93225262, 2.41593945);
  let x1 = vec3f(0.69549072, 0.49228336, 0.27699880);
  let y1 = vec3f(0.02312639, 0.15225084, 0.52607955);
  let c2 = vec3f(3.90307140, 3.21182957, 3.96587128);
  let x2 = vec3f(0.11748627, 0.86755042, 0.66077860);
  let y2 = vec3f(0.84897130, 0.88445281, 0.73949448);
  return bump3(c1 * (x - x1), y1) + bump3(c2 * (x - x2), y2);
}

/// Reflectance tint of a thin film (thickness in nm, refractive index `ior`) over
/// a reflective base, seen at cos(θ) = `cos_v`. Two-beam interference: the film's
/// top and bottom reflections interfere with an optical path difference
/// 2·n·d·cos θt (+ half-wave flip), evaluated at three representative wavelengths.
fn thin_film(cos_v: f32, thickness: f32, ior: f32) -> vec3f {
  let sin_t2 = (1.0 - cos_v * cos_v) / (ior * ior);
  let cos_t = sqrt(max(1.0 - sin_t2, 0.0));
  let opd = 2.0 * ior * thickness * cos_t;
  let lambda = vec3f(650.0, 532.0, 450.0);
  let phase = TAU * opd / lambda + PI;
  return 0.5 + 0.5 * cos(phase);
}

/// Reflective diffraction grating with grooves perpendicular to `t` (unit, in the
/// surface plane) and groove spacing `spacing` nm. Light from `l` diffracts toward
/// `v` for wavelengths λ = spacing·|t·(l+v)|/m (grating equation, orders 1..3),
/// concentrated near the plane where b·(l+v) ≈ 0 (b = n × t).
fn diffraction_grating(l: vec3f, v: vec3f, n: vec3f, t: vec3f, spacing: f32, lobe_width: f32) -> vec3f {
  let h = l + v;
  let b = cross(n, t);
  let u = abs(dot(t, h));
  let off_plane = dot(b, h) / max(lobe_width, 1e-3);
  let lobe = exp(-off_plane * off_plane) * saturate(dot(n, l)) * saturate(dot(n, v) * 4.0);
  var c = vec3f(0.0);
  for (var m = 1; m <= 3; m++) {
    let w = spacing * u / f32(m);
    c += spectral_rgb(w) * step(380.0, w) * step(w, 720.0) / f32(m);
  }
  return c * lobe;
}
