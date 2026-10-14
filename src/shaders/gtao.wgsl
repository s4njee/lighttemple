// ---------------------------------------------------------------------------
// gtao.wgsl — screen-space ambient occlusion kernels and their shading terms.
//
//   gtao(px, p)     Ground-truth AO (Jimenez et al. 2016), structured after
//                   Intel's XeGTAO: per pixel, a few *slices* (screen-space
//                   directions through the pixel); in each, march both ways
//                   to find the highest horizon on either side, clamp the two
//                   horizons to the normal's hemisphere and integrate the
//                   cosine-weighted visible arc between them analytically.
//                   Also returns the bent normal (the average unoccluded
//                   direction) from the same integral.
//   ssao(px, p)     Classic normal-oriented hemisphere SSAO, for comparison:
//                   random points in a hemisphere, each tested against the
//                   depth buffer, with a range check.
//   gtao_multibounce(ao, albedo)   Jimenez's fit for light that bounces
//                   inside the occluded region (bright albedos darken less).
//   specular_occlusion(nov, ao, roughness)   Lagarde & de Rousiers 2014.
//
// Requires (prepended): common.wgsl, screen-space.wgsl, and this binding
// declared by the including shader (see GBuffer.resolve):
//   var ss_geometry: texture_2d<f32>;   // rgba32float, view normal xyz + depth w
// ---------------------------------------------------------------------------

struct AoParams {
  proj_scale: vec2f,
  jitter: vec2f,
  /// World-space radius of the occlusion search.
  radius: f32,
  /// GTAO: fraction of the radius over which distant occluders fade out.
  falloff: f32,
  /// Cap on the search radius in pixels of the geometry texture (cost and cache misses).
  max_radius_px: f32,
  /// GTAO: slices per pixel and steps per side of each slice. SSAO: samples (`steps`).
  slices: i32,
  steps: i32,
  /// SSAO: depth bias in world units.
  bias: f32,
  /// Per-pixel noise in [0, 1): slice rotation and step offset.
  noise_slice: f32,
  noise_step: f32,
};

struct AoResult {
  /// View-space bent normal, *not* normalised: its length is 1 for an open surface and shrinks
  /// as the unoccluded cone narrows. Average it linearly (it is an integral), normalise to use it.
  bent: vec3f,
  /// Ambient visibility, 1 = unoccluded.
  visibility: f32,
};

/// The geometry texel at `px`, or w = 0 (no geometry) off-screen.
fn ss_geometry_at(px: vec2i) -> vec4f {
  let dims = vec2i(textureDimensions(ss_geometry));
  if (any(px < vec2i(0)) || any(px >= dims)) { return vec4f(0.0); }
  return textureLoad(ss_geometry, px, 0);
}

fn gtao(px: vec2i, p: AoParams) -> AoResult {
  let dims = vec2f(textureDimensions(ss_geometry));
  let g = textureLoad(ss_geometry, px, 0);
  let n = g.xyz;
  if (g.w <= 0.0) { return AoResult(n, 1.0); }
  let pos = ss_view_pos((vec2f(px) + 0.5) / dims, g.w, p.proj_scale, p.jitter);
  let v = normalize(-pos);

  // World radius -> pixels at this depth.
  let radius_px = min(p.radius * p.proj_scale.y * 0.5 * dims.y / g.w, p.max_radius_px);
  if (radius_px < 1.0) { return AoResult(n, 1.0); }
  // Occluders fade out linearly over the last `falloff` of the radius (no hard cut-off ring).
  let falloff_range = max(p.falloff * p.radius, 1e-4);
  let falloff_mul = -1.0 / falloff_range;
  let falloff_add = p.radius / falloff_range;
  // Start ~1.4 px out, so the first step never lands on this pixel.
  let min_s = min(1.42 / radius_px, 1.0);

  var visibility = 0.0;
  var bent = vec3f(0.0);
  for (var s = 0; s < p.slices; s++) {
    let phi = (f32(s) + p.noise_slice) / f32(p.slices) * PI;
    let omega = vec2f(cos(phi), sin(phi));
    // The slice plane holds v and the screen direction (y flips: screen y is down, view y up).
    let dir = vec3f(omega.x, -omega.y, 0.0);
    let ortho = dir - dot(dir, v) * v;
    let axis = normalize(cross(dir, v));
    let proj_n = n - axis * dot(n, axis);
    let proj_len = length(proj_n);
    let cos_n = saturate(dot(proj_n, v) / max(proj_len, 1e-5));
    // Angle of the projected normal from v, positive toward the +omega side.
    let nn = sign(dot(ortho, proj_n)) * acos(cos_n);

    // Horizon cosines, starting flat on the tangent plane (nothing occluded).
    let low0 = cos(nn + 0.5 * PI);
    let low1 = cos(nn - 0.5 * PI);
    var hc0 = low0;
    var hc1 = low1;
    for (var j = 0; j < p.steps; j++) {
      // Quadratic spacing: dense near the pixel, where contact occlusion lives.
      var t = (f32(j) + p.noise_step) / f32(p.steps);
      t = mix(min_s, 1.0, t * t);
      let off = vec2i(round(omega * (t * radius_px)));
      let g0 = ss_geometry_at(px + off);
      let g1 = ss_geometry_at(px - off);
      if (g0.w > 0.0) {
        let d = ss_view_pos((vec2f(px + off) + 0.5) / dims, g0.w, p.proj_scale, p.jitter) - pos;
        let len = length(d);
        let w = saturate(len * falloff_mul + falloff_add);
        hc0 = max(hc0, mix(low0, dot(d, v) / len, w));
      }
      if (g1.w > 0.0) {
        let d = ss_view_pos((vec2f(px - off) + 0.5) / dims, g1.w, p.proj_scale, p.jitter) - pos;
        let len = length(d);
        let w = saturate(len * falloff_mul + falloff_add);
        hc1 = max(hc1, mix(low1, dot(d, v) / len, w));
      }
    }

    // Horizon angles (h0 on the -omega side, negative), clamped to the normal's hemisphere.
    let h0 = nn + max(-acos(clamp(hc1, -1.0, 1.0)) - nn, -0.5 * PI);
    let h1 = nn + min(acos(clamp(hc0, -1.0, 1.0)) - nn, 0.5 * PI);
    let sin_n = sin(nn);
    // Cosine-weighted visible arc between the horizons, in closed form.
    let arc0 = (cos_n + 2.0 * h0 * sin_n - cos(2.0 * h0 - nn)) * 0.25;
    let arc1 = (cos_n + 2.0 * h1 * sin_n - cos(2.0 * h1 - nn)) * 0.25;
    visibility += proj_len * (arc0 + arc1);

    // Bent normal: the same integral weighted by direction (Algorithm 2 of the paper).
    let t0 = (6.0 * sin(h0 - nn) - sin(3.0 * h0 - nn) + 6.0 * sin(h1 - nn) - sin(3.0 * h1 - nn) + 16.0 * sin_n
      - 3.0 * (sin(h0 + nn) + sin(h1 + nn))) / 12.0;
    let t1 = (-cos(3.0 * h0 - nn) - cos(3.0 * h1 - nn) + 8.0 * cos(nn) - 3.0 * (cos(h0 + nn) + cos(h1 + nn))) / 12.0;
    bent += (normalize(ortho) * t0 + v * t1) * proj_len;
  }
  // Unnormalised, so filters can average it linearly; scaled so an open surface returns its normal.
  return AoResult(bent * (1.5 / f32(p.slices)), saturate(visibility / f32(p.slices)));
}

fn ssao(px: vec2i, p: AoParams) -> AoResult {
  let dims = vec2f(textureDimensions(ss_geometry));
  let g = textureLoad(ss_geometry, px, 0);
  let n = g.xyz;
  if (g.w <= 0.0) { return AoResult(n, 1.0); }
  let pos = ss_view_pos((vec2f(px) + 0.5) / dims, g.w, p.proj_scale, p.jitter);

  // A tangent frame around n, spun per pixel by the noise.
  let up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.9);
  let t0 = normalize(cross(up, n));
  let b0 = cross(n, t0);
  let a = p.noise_slice * TAU;
  let t = t0 * cos(a) + b0 * sin(a);
  let b = cross(n, t);

  let count = max(p.steps, 1);
  var occlusion = 0.0;
  var bent = vec3f(0.0);
  for (var i = 0; i < count; i++) {
    // Cosine-distributed directions on a Fibonacci spiral, at radii biased toward the centre.
    let u = (f32(i) + 0.5) / f32(count);
    let ang = f32(i) * 2.39996323;
    let r = sqrt(u);
    let dir = t * (r * cos(ang)) + b * (r * sin(ang)) + n * sqrt(1.0 - u);
    let k = fract(u * 7.0 + p.noise_step);
    let sp = pos + dir * (p.radius * mix(0.1, 1.0, k * k));
    if (sp.z >= -1e-3) { bent += dir; continue; }
    let suv = ss_view_uv(sp, p.proj_scale, p.jitter);
    let sz = ss_geometry_at(vec2i(floor(suv * dims))).w;
    // Off-screen or background: unoccluded. Otherwise occluded if the stored surface is in front.
    if (sz <= 0.0 || sz > -sp.z - p.bias) { bent += dir; continue; }
    // Range check: an occluder much nearer the camera than this pixel is a different object.
    occlusion += smoothstep(0.0, 1.0, p.radius / abs(g.w - sz));
  }
  // Mean unoccluded direction; cosine-distributed directions average to (2/3)·n in the open.
  return AoResult(bent * (1.5 / f32(count)), 1.0 - occlusion / f32(count));
}

/// Jimenez et al. 2016: AO that accounts for interreflections inside the occluded region.
fn gtao_multibounce(ao: f32, albedo: vec3f) -> vec3f {
  let a = 2.0404 * albedo - 0.3324;
  let b = -4.7951 * albedo + 0.6417;
  let c = 2.7552 * albedo + 0.6903;
  return max(vec3f(ao), ((ao * a + b) * ao + c) * ao);
}

/// Lagarde & de Rousiers 2014: specular occlusion from ambient occlusion, sharper for glossy lobes.
fn specular_occlusion(nov: f32, ao: f32, roughness: f32) -> f32 {
  return saturate(pow(nov + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao);
}
