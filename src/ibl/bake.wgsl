// ---------------------------------------------------------------------------
// bake.wgsl — one-time image-based-lighting precomputation.
//
//   fs_equirect / fs_studio / fs_sky   source -> environment cubemap (mip 0)
//   fs_downsample                      environment mip chain (box filter)
//   fs_prefilter                       GGX-prefiltered specular cube, one mip per roughness
//   cs_sh                              9-coefficient spherical-harmonic irradiance
//   fs_brdf_lut                        split-sum DFG lookup table
//
// Cube faces are rendered one layer at a time; the instance index carries the
// face (and, for the prefilter, the mip) so no per-pass uniforms are needed.
// Requires (prepended): common.wgsl, brdf.wgsl, the `Bake` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var src_equirect: texture_2d<f32>;
@group(0) @binding(1) var bake_sampler: sampler;
@group(0) @binding(2) var<uniform> bake: Bake;
@group(0) @binding(3) var src_cube: texture_cube<f32>;
@group(0) @binding(4) var<storage, read_write> sh_out: array<vec4f, 9>;

struct BakeOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) id: u32,
};

@vertex
fn vs_bake(@builtin(vertex_index) i: u32, @builtin(instance_index) id: u32) -> BakeOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var out: BakeOut;
  out.position = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(p.x, 1.0 - p.y);
  out.id = id;
  return out;
}

/// Direction through texel `uv` (origin top-left) of cube face `face`
/// (+X, -X, +Y, -Y, +Z, -Z), matching WebGPU cube sampling.
fn cube_dir(face: u32, uv: vec2f) -> vec3f {
  let s = uv.x * 2.0 - 1.0;
  let t = uv.y * 2.0 - 1.0;
  switch face {
    case 0u: { return normalize(vec3f(1.0, -t, -s)); }
    case 1u: { return normalize(vec3f(-1.0, -t, s)); }
    case 2u: { return normalize(vec3f(s, 1.0, t)); }
    case 3u: { return normalize(vec3f(s, -1.0, -t)); }
    case 4u: { return normalize(vec3f(s, -t, 1.0)); }
    default: { return normalize(vec3f(-s, -t, -1.0)); }
  }
}

// --- Sources -------------------------------------------------------------------

@fragment
fn fs_equirect(v: BakeOut) -> @location(0) vec4f {
  let d = cube_dir(v.id, v.uv);
  let uv = vec2f(atan2(d.x, -d.z) / TAU + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
  return vec4f(textureSampleLevel(src_equirect, bake_sampler, uv, 0.0).rgb, 1.0);
}

/// A rectangular softbox facing the origin: `c` = direction to its centre,
/// `half_size` = half extents on the plane at distance 1 (≈ tan of half-angles).
fn softbox(d: vec3f, c: vec3f, half_size: vec2f, radiance: vec3f) -> vec3f {
  let k = dot(d, c);
  if (k <= 0.0) { return vec3f(0.0); }
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), c) + vec3f(1e-5, 0.0, 0.0));
  let up = cross(c, right);
  let p = d / k - c;
  let q = abs(vec2f(dot(p, right), dot(p, up))) / half_size;
  let edge = 0.06;
  let mask = (1.0 - smoothstep(1.0 - edge, 1.0 + edge, q.x)) * (1.0 - smoothstep(1.0 - edge, 1.0 + edge, q.y));
  // Real softboxes are brighter in the middle than at the diffuser's edges.
  let falloff = 1.0 - 0.35 * dot(q, q) * 0.5;
  return radiance * mask * falloff;
}

@fragment
fn fs_studio(v: BakeOut) -> @location(0) vec4f {
  let d = cube_dir(v.id, v.uv);
  // Dark seamless backdrop: slightly lighter towards the horizon, darker floor.
  var c = mix(vec3f(0.018, 0.017, 0.016), vec3f(0.05, 0.048, 0.045), 1.0 - abs(d.y));
  c = select(c, vec3f(0.03, 0.028, 0.026) * (1.0 + d.y), d.y < 0.0);
  c += softbox(d, normalize(vec3f(-0.55, 0.65, 0.55)), vec2f(0.55, 0.38), vec3f(9.0, 8.4, 7.6));   // key
  c += softbox(d, normalize(vec3f(0.85, 0.2, 0.35)), vec2f(0.22, 0.55), vec3f(1.6, 1.8, 2.1));    // fill
  c += softbox(d, normalize(vec3f(0.15, 0.55, -0.85)), vec2f(0.75, 0.08), vec3f(7.0, 7.0, 7.0));  // rim strip
  c += softbox(d, vec3f(0.0, 1.0, 0.0), vec2f(0.35, 0.35), vec3f(1.2, 1.2, 1.2));                // top bounce
  return vec4f(c * bake.exposure, 1.0);
}

@fragment
fn fs_sky(v: BakeOut) -> @location(0) vec4f {
  let d = cube_dir(v.id, v.uv);
  let sun = normalize(bake.sunDir);
  // Warmer, dimmer light as the sun approaches the horizon.
  let low = 1.0 - smoothstep(0.0, 0.45, sun.y);
  let zenith = mix(vec3f(0.18, 0.32, 0.65), vec3f(0.12, 0.14, 0.32), low);
  let horizon = mix(vec3f(0.75, 0.82, 0.92), vec3f(1.1, 0.55, 0.28), low);
  let sun_col = mix(vec3f(1.0, 0.95, 0.88), vec3f(1.0, 0.45, 0.18), low) * mix(1.0, 0.45, low);

  let h = max(d.y, 0.0);
  var c = mix(horizon, zenith, pow(h, 0.45));
  let mu = dot(d, sun);
  // Mie-ish forward glow around the sun, broad and tight lobes.
  c += sun_col * (0.35 * pow(saturate(mu), 8.0) + 1.2 * pow(saturate(mu), 120.0));
  // Sun disc (angular radius ~1.3°). Its total irradiance is ~3x the sky's, as
  // outdoors, so it dominates the SH band-1 vector and casts a real shadow;
  // filtered importance sampling keeps the prefiltered cube free of fireflies.
  c += sun_col * 2500.0 * smoothstep(cos(0.024), cos(0.020), mu) * step(0.0, d.y + 0.02);
  // Ground: dark warm albedo lit by the sky, with horizon haze.
  let ground = vec3f(0.11, 0.095, 0.08) * (0.3 + 0.7 * saturate(sun.y + 0.2));
  c = select(c, mix(horizon * 0.6, ground, saturate(-d.y * 6.0)), d.y < 0.0);
  return vec4f(c * bake.exposure, 1.0);
}

// --- Mip chain -----------------------------------------------------------------

@fragment
fn fs_downsample(v: BakeOut) -> @location(0) vec4f {
  // Destination texel centres sit between 4 source texels: one bilinear tap = 2x2 box.
  return vec4f(textureSampleLevel(src_cube, bake_sampler, cube_dir(v.id, v.uv), 0.0).rgb, 1.0);
}

// --- GGX importance sampling -----------------------------------------------------

fn hammersley(i: u32, n: u32) -> vec2f {
  return vec2f(f32(i) / f32(n), f32(reverseBits(i)) * 2.3283064365386963e-10);
}

fn importance_ggx(xi: vec2f, n: vec3f, a: f32) -> vec3f {
  let phi = TAU * xi.x;
  let cos_t = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
  let sin_t = sqrt(1.0 - cos_t * cos_t);
  let h = vec3f(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
  let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
  let tx = normalize(cross(up, n));
  let ty = cross(n, tx);
  return normalize(tx * h.x + ty * h.y + n * h.z);
}

// Filtered importance sampling (Křivánek & Colbert 2008): read a blurrier mip for
// low-probability samples, so 64 samples look like thousands.
@fragment
fn fs_prefilter(v: BakeOut) -> @location(0) vec4f {
  let face = v.id % 6u;
  let mip = f32(v.id / 6u);
  let roughness = mip / max(bake.mipCount - 1.0, 1.0);
  let n = cube_dir(face, v.uv);
  if (roughness == 0.0) {
    return vec4f(textureSampleLevel(src_cube, bake_sampler, n, 0.0).rgb, 1.0);
  }
  let a = roughness * roughness;
  let texel_solid_angle = 4.0 * PI / (6.0 * bake.envSize * bake.envSize);
  var sum = vec3f(0.0);
  var weight = 0.0;
  for (var i = 0u; i < bake.sampleCount; i++) {
    let h = importance_ggx(hammersley(i, bake.sampleCount), n, a);
    let l = normalize(2.0 * dot(n, h) * h - n);
    let nol = dot(n, l);
    if (nol > 0.0) {
      let noh = saturate(dot(n, h));
      // With N = V, pdf(L) = D(h) * NoH / (4 VoH) = D / 4.
      let pdf = d_ggx(noh, a) * 0.25;
      let sample_solid_angle = 1.0 / (f32(bake.sampleCount) * pdf + 1e-4);
      let lod = max(0.5 * log2(sample_solid_angle / texel_solid_angle) + 1.0, 0.0);
      sum += textureSampleLevel(src_cube, bake_sampler, l, lod).rgb * nol;
      weight += nol;
    }
  }
  return vec4f(sum / max(weight, 1e-4), 1.0);
}

// --- Spherical harmonics ----------------------------------------------------------

fn sh_basis(d: vec3f) -> array<f32, 9> {
  return array<f32, 9>(
    0.282095,
    0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x,
    1.092548 * d.x * d.y, 1.092548 * d.y * d.z, 0.315392 * (3.0 * d.z * d.z - 1.0),
    1.092548 * d.x * d.z, 0.546274 * (d.x * d.x - d.y * d.y));
}

var<workgroup> sh_acc: array<array<vec3f, 9>, 64>;

@compute @workgroup_size(64)
fn cs_sh(@builtin(local_invocation_index) li: u32) {
  let size = bake.shSize;
  let per_face = size * size;
  var acc: array<vec3f, 9>;
  for (var i = li; i < 6u * per_face; i += 64u) {
    let face = i / per_face;
    let rem = i % per_face;
    let uv = (vec2f(f32(rem % size), f32(rem / size)) + 0.5) / f32(size);
    let st = uv * 2.0 - 1.0;
    // Solid angle of this texel on the unit cube.
    let dw = (4.0 / f32(per_face)) / pow(1.0 + dot(st, st), 1.5);
    let d = cube_dir(face, uv);
    let c = textureSampleLevel(src_cube, bake_sampler, d, bake.shLod).rgb * dw;
    let y = sh_basis(d);
    for (var k = 0; k < 9; k++) { acc[k] += c * y[k]; }
  }
  sh_acc[li] = acc;
  workgroupBarrier();
  if (li == 0u) {
    // Fold in the clamped-cosine convolution (Ramamoorthi & Hanrahan) so shading
    // just evaluates sum(c_k * Y_k(n)) to get irradiance E(n).
    let band = array<f32, 9>(PI, 2.0 * PI / 3.0, 2.0 * PI / 3.0, 2.0 * PI / 3.0, PI / 4.0, PI / 4.0, PI / 4.0, PI / 4.0, PI / 4.0);
    for (var k = 0; k < 9; k++) {
      var total = vec3f(0.0);
      for (var t = 0; t < 64; t++) { total += sh_acc[t][k]; }
      sh_out[k] = vec4f(total * band[k], 0.0);
    }
  }
}

// --- Split-sum DFG lookup table -------------------------------------------------

@fragment
fn fs_brdf_lut(v: BakeOut) -> @location(0) vec4f {
  let nov = max(v.uv.x, 1e-3);
  let roughness = v.uv.y;
  let a = max(roughness * roughness, 1e-3);
  let vdir = vec3f(sqrt(1.0 - nov * nov), 0.0, nov);
  let n = vec3f(0.0, 0.0, 1.0);
  var r = vec2f(0.0);
  let count = 512u;
  for (var i = 0u; i < count; i++) {
    let h = importance_ggx(hammersley(i, count), n, a);
    let l = 2.0 * dot(vdir, h) * h - vdir;
    let nol = saturate(l.z);
    let noh = saturate(h.z);
    let voh = saturate(dot(vdir, h));
    if (nol > 0.0) {
      let g = v_smith_ggx_correlated(nov, nol, a) * voh * nol / max(noh, 1e-4);
      let fc = pow(1.0 - voh, 5.0);
      r += vec2f(g * (1.0 - fc), g * fc);
    }
  }
  return vec4f(r * 4.0 / f32(count), 0.0, 1.0);
}
