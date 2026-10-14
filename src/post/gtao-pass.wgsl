// ---------------------------------------------------------------------------
// gtao-pass.wgsl — one fullscreen pass at the geometry texture's resolution
// running GTAO or classic SSAO per pixel (kernels in src/shaders/gtao.wgsl).
// Output: rgba16float, view-space bent normal xyz (unnormalised, see AoResult)
// + visibility w.
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, screen-space.wgsl, the
// `Gtao` struct, then src/shaders/gtao.wgsl.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> ao: Gtao;
@group(0) @binding(1) var ss_geometry: texture_2d<f32>;

@fragment
fn fs_ao(v: FullscreenOut) -> @location(0) vec4f {
  let px = vec2i(v.position.xy);
  let fp = vec2f(px);
  var p: AoParams;
  p.proj_scale = ao.projScale;
  p.jitter = ao.jitter;
  p.radius = ao.radius;
  p.falloff = ao.falloff;
  p.max_radius_px = ao.maxRadiusPx;
  p.slices = i32(ao.slices);
  p.steps = i32(ao.steps);
  p.bias = ao.bias;
  // Two decorrelated per-pixel noises, shifted every frame so temporal accumulation sees new samples.
  p.noise_slice = ign(fp + ao.noiseOffset);
  p.noise_step = fract(hash12(fp) + ao.noiseOffset * 0.61803399);
  var r: AoResult;
  if (ao.method == 1u) { r = ssao(px, p); } else { r = gtao(px, p); }
  return vec4f(r.bent, r.visibility);
}
