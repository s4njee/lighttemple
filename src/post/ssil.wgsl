// ---------------------------------------------------------------------------
// ssil.wgsl — screen-space indirect lighting with a visibility bitmask
// (Therrien, Levesque & Gilet 2023), one fullscreen pass at the geometry
// texture's resolution.
//
// It walks the same slices as GTAO, but instead of keeping only the highest
// horizon on each side it keeps a 32-bit mask of which *sectors* of the
// slice's hemisphere are blocked. Each sample is a surface `thickness` deep:
// it blocks the sectors between the angles of its front face and of its back
// face, so light can pass behind thin objects (GTAO assumes every occluder
// extends infinitely far back). The sectors a sample blocks that nothing
// nearer has already blocked are exactly the directions in which this pixel
// *sees* that sample, so they are also how much of its light arrives: the
// sample's radiance (from a lit buffer) times the share of fresh sectors,
// times the cosine at the receiver.
//
// Output: rgba16float, rgb = incoming indirect light normalised so that
// albedo × rgb is the bounced radiance (a uniform environment of radiance L
// gives L), a = visibility (1 − blocked sectors), an AO with thickness.
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, screen-space.wgsl and
// the `Ssil` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> ssil: Ssil;
@group(0) @binding(1) var ss_geometry: texture_2d<f32>;
// Lit colour with a mip chain: distant samples read prefiltered radiance.
@group(0) @binding(2) var radiance: texture_2d<f32>;
@group(0) @binding(3) var radiance_smp: sampler;

const SECTORS: f32 = 32.0;

/// Bits for the sectors spanning [lo, hi] of the hemisphere (0..1 across it).
fn sector_bits(lo: f32, hi: f32) -> u32 {
  let a = saturate(lo);
  let b = saturate(hi);
  let count = u32(ceil((b - a) * SECTORS));
  if (count == 0u) { return 0u; }
  let start = min(u32(a * SECTORS), 31u);
  return (0xffffffffu >> (32u - count)) << start;
}

@fragment
fn fs_ssil(frag: FullscreenOut) -> @location(0) vec4f {
  let px = vec2i(frag.position.xy);
  let idims = vec2i(textureDimensions(ss_geometry));
  let dims = vec2f(idims);
  let g = textureLoad(ss_geometry, px, 0);
  if (g.w <= 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let n = g.xyz;
  let pos = ss_view_pos((vec2f(px) + 0.5) / dims, g.w, ssil.projScale, ssil.jitter);
  let v = normalize(-pos);
  let radius_px = min(ssil.radius * ssil.projScale.y * 0.5 * dims.y / g.w, ssil.maxRadiusPx);
  if (radius_px < 1.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let min_s = min(1.42 / radius_px, 1.0);
  let steps = i32(ssil.steps);
  let slices = i32(ssil.slices);
  let fp = vec2f(px);
  let noise_slice = ign(fp + ssil.noiseOffset);
  let noise_step = fract(hash12(fp) + ssil.noiseOffset * 0.61803399);

  var gi = vec3f(0.0);
  var visibility = 0.0;
  for (var s = 0; s < slices; s++) {
    let phi = (f32(s) + noise_slice) / f32(slices) * PI;
    let omega = vec2f(cos(phi), sin(phi));
    let dir = vec3f(omega.x, -omega.y, 0.0);
    let ortho = dir - dot(dir, v) * v;
    let axis = normalize(cross(dir, v));
    let proj_n = n - axis * dot(n, axis);
    let cos_n = saturate(dot(proj_n, v) / max(length(proj_n), 1e-5));
    let nn = sign(dot(ortho, proj_n)) * acos(cos_n);

    var occluded = 0u;
    for (var side = 0; side < 2; side++) {
      let sgn = select(1.0, -1.0, side == 1);
      for (var j = 0; j < steps; j++) {
        var t = (f32(j) + noise_step) / f32(steps);
        t = mix(min_s, 1.0, t * t);
        let q = px + vec2i(round(omega * (sgn * t * radius_px)));
        if (any(q < vec2i(0)) || any(q >= idims)) { break; }
        let sg = textureLoad(ss_geometry, q, 0);
        if (sg.w <= 0.0) { continue; }
        let suv = (vec2f(q) + 0.5) / dims;
        let delta = ss_view_pos(suv, sg.w, ssil.projScale, ssil.jitter) - pos;
        let dist = length(delta);
        if (dist < 1e-4) { continue; }
        let front = delta / dist;
        let back = normalize(delta - v * ssil.thickness);
        // Signed angles from v (positive toward +omega), then 0..1 across the hemisphere around the normal.
        let a_front = sgn * acos(clamp(dot(front, v), -1.0, 1.0));
        let a_back = sgn * acos(clamp(dot(back, v), -1.0, 1.0));
        let t_front = (a_front - nn) / PI + 0.5;
        let t_back = (a_back - nn) / PI + 0.5;
        let bits = sector_bits(min(t_front, t_back), max(t_front, t_back));
        let fresh = bits & ~occluded;
        if (fresh != 0u) {
          // Footprint of this step in radiance texels picks the prefiltered level.
          let spacing = 2.0 * t * radius_px * ssil.radianceScale / f32(steps);
          let lod = clamp(log2(max(spacing, 1.0)) - 1.0, 0.0, ssil.maxLod);
          let l = textureSampleLevel(radiance, radiance_smp, suv, lod).rgb;
          // Lit buffers hold the radiance of the side we see; samples facing away emit nothing toward us.
          let facing = smoothstep(-0.1, 0.2, dot(sg.xyz, -front));
          gi += f32(countOneBits(fresh)) / SECTORS * l * saturate(dot(n, front)) * facing;
        }
        occluded |= bits;
      }
    }
    visibility += 1.0 - f32(countOneBits(occluded)) / SECTORS;
  }
  // Uniform sectors weighted by cos average 2/π over a full hemisphere: scale so that L in, L out.
  return vec4f(gi * (0.5 * PI / f32(slices)), visibility / f32(slices));
}
