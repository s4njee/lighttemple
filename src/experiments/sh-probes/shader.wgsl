// ---------------------------------------------------------------------------
// Irradiance probes: a 3D grid of SH probes baked on the GPU by ray casting the
// room analytically, sampled with trilinear interpolation and DDGI-style leak
// reduction (normal bias, back-face weights, Chebyshev depth visibility).
// Outputs premultiplied linear HDR into the post chain.
// Requires (prepended): common.wgsl, noise.wgsl, sh.wgsl, brdf.wgsl,
// shadow.wgsl, sun-room.wgsl, VertexIn, PROP_WGSL, ROOM_PRIM_WGSL,
// sun-room-trace.wgsl, and the `Frame` and `Step` structs.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> prims: array<Prim>;
// Probe visibility (mean distance, mean squared distance) in 10×10 octahedral tiles,
// and one probe volume: 9 irradiance coefficients per probe (coefficient 0's w = alive).
@group(1) @binding(0) var vis_atlas: texture_2d<f32>;
@group(1) @binding(1) var lin_smp: sampler;
@group(1) @binding(2) var<storage, read> probes: array<vec4f>;
// Bake (compute) only.
@group(2) @binding(0) var<uniform> bake: Step;
@group(2) @binding(1) var<storage, read_write> probes_out: array<vec4f>;
@group(2) @binding(2) var atlas_out: texture_storage_2d<rgba16float, write>;
// Camera passes only.
@group(2) @binding(3) var<storage, read> objects: array<Object>;
@group(2) @binding(4) var shadow_map: texture_depth_2d;
@group(2) @binding(5) var shadow_cmp: sampler_comparison;

/// Interior texels per side of a probe's octahedral visibility tile (plus a 1-texel border).
const OCT = 8u;
const TILE = 10u;
/// Hit distances are clamped here (rays leaving through the window never return).
const MAX_DIST = 6.0;
/// Rays per probe for the visibility map. It is baked once per grid, so it can afford many more
/// rays than the lighting, and a sharper filter lobe.
const VIS_RAYS = 512u;
const VIS_SHARPNESS = 24.0;

const LEAK_OFF = 0u;
const LEAK_VISIBILITY = 2u;

// --- Probe grid ---------------------------------------------------------------

fn probe_coord(i: u32) -> vec3u {
  let d = frame.gridDims.xyz;
  return vec3u(i % d.x, (i / d.x) % d.y, i / (d.x * d.y));
}

fn probe_index(c: vec3u) -> u32 {
  let d = frame.gridDims.xyz;
  return c.x + d.x * (c.y + d.y * c.z);
}

fn probe_pos(c: vec3u) -> vec3f {
  return frame.gridMin + (vec3f(c) + 0.5) * frame.gridCell;
}

/// Top-left texel of probe c's tile in the visibility atlas.
fn probe_tile(c: vec3u) -> vec2u {
  return vec2u(c.x + frame.gridDims.x * c.z, c.y) * TILE;
}

// --- Octahedral mapping (Cigolle et al. 2014) -----------------------------------

fn sign_not_zero(v: vec2f) -> vec2f {
  return select(vec2f(-1.0), vec2f(1.0), v >= vec2f(0.0));
}

fn oct_encode(d: vec3f) -> vec2f {
  let p = d.xy / (abs(d.x) + abs(d.y) + abs(d.z));
  return select(p, (1.0 - abs(p.yx)) * sign_not_zero(p), d.z < 0.0);
}

fn oct_decode(e: vec2f) -> vec3f {
  var v = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) { v = vec3f((1.0 - abs(v.yx)) * sign_not_zero(v.xy), v.z); }
  return normalize(v);
}

/// Mean and mean-squared distance seen by probe c in direction d (bilinear, seamless thanks to the border).
fn probe_moments(c: vec3u, d: vec3f) -> vec2f {
  let px = vec2f(probe_tile(c)) + 1.0 + (oct_encode(d) * 0.5 + 0.5) * f32(OCT);
  return textureSampleLevel(vis_atlas, lin_smp, px / frame.atlasSize, 0.0).xy;
}

// --- Sampling the volume ----------------------------------------------------------

struct ProbeLookup {
  sh: array<vec3f, 9>,
  /// Share of the trilinear weight that survived the visibility test (1 = every probe trusted).
  trust: f32,
};

/// Blends the eight probes around p into one set of SH coefficients. `bias_dir` pushes the
/// lookup off the surface (the normal, leaning toward the viewer in camera passes).
fn sample_probes(p: vec3f, n: vec3f, bias_dir: vec3f) -> ProbeLookup {
  let leak = frame.leakMode;
  // The camera looks in from outside the room, so a bias toward it can cross a cut-away wall:
  // keep the biased point inside.
  let q = select(room_clamp_inside(p + bias_dir * frame.normalBias, 0.02), p, leak == LEAK_OFF);
  let g = (q - frame.gridMin) / frame.gridCell - 0.5;
  let base = clamp(floor(g), vec3f(0.0), vec3f(frame.gridDims.xyz) - 2.0);
  let a = clamp(g - base, vec3f(0.0), vec3f(1.0));
  var out: ProbeLookup;
  var wsum = 0.0;
  var tsum = 0.0;
  var vsum = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let o = vec3u(i & 1u, (i >> 1u) & 1u, i >> 2u);
    let c = vec3u(base) + o;
    let idx = probe_index(c) * 9u;
    let alive = probes[idx].w;
    let t3 = select(1.0 - a, a, o == vec3u(1u));
    let tri = t3.x * t3.y * t3.z * alive;
    var w = 1.0;
    var visible = 1.0;
    if (leak != LEAK_OFF) {
      let pp = probe_pos(c);
      // Back-face weight: probes behind the surface see its other side.
      let to_probe = normalize(pp - p);
      let back = (dot(to_probe, n) + 1.0) * 0.5;
      w = back * back + 0.2;
      if (leak == LEAK_VISIBILITY) {
        // Chebyshev: how likely is the probe to see this point, given the distances it recorded
        // in this direction? Anything farther than its mean hit is probably behind a wall.
        let v = q - pp;
        let dist = length(v);
        let m = probe_moments(c, v / max(dist, 1e-4));
        if (dist > m.x) {
          let variance = abs(m.y - m.x * m.x) + 1e-4;
          let excess = dist - m.x;
          let cheb = variance / (variance + excess * excess);
          visible = cheb * cheb * cheb;
        }
      }
      // Crush tiny weights so a barely-visible probe cannot dominate (DDGI).
      w = max(w * visible, 1e-6);
      if (w < 0.2) { w *= w * w * 25.0; }
    }
    w *= tri;
    for (var k = 0u; k < 9u; k++) { out.sh[k] += probes[idx + k].rgb * w; }
    wsum += w;
    tsum += tri;
    vsum += tri * visible;
  }
  let inv = select(0.0, 1.0 / wsum, wsum > 1e-7);
  for (var k = 0u; k < 9u; k++) { out.sh[k] *= inv; }
  out.trust = select(0.0, vsum / tsum, tsum > 1e-7);
  return out;
}

fn probe_irradiance(l: ProbeLookup, n: vec3f) -> vec3f {
  return sh9_irradiance(l.sh, n, frame.shOrder, frame.dering);
}

// --- Bake ---------------------------------------------------------------------------

/// Direction i of n, spherical Fibonacci: near-uniform, with no clumps or gaps.
fn ray_dir(i: u32, n: u32) -> vec3f {
  let z = 1.0 - (2.0 * f32(i) + 1.0) / f32(n);
  let r = sqrt(max(0.0, 1.0 - z * z));
  let phi = f32(i) * 2.39996323;
  return vec3f(r * cos(phi), r * sin(phi), z);
}

fn outdoor_radiance(d: vec3f) -> vec3f {
  return room_outdoor(d, frame.skyZenith.rgb, frame.skyHorizon.rgb, frame.outdoorGround.rgb);
}

/// Radiance arriving at a probe from direction d. `bounce` > 1 adds the previous volume's
/// irradiance at the hit point, i.e. one more bounce.
fn trace_radiance(p: vec3f, d: vec3f, bounce: u32) -> vec3f {
  let h = room_trace(p, d, 1e-4, 100.0);
  if (h.t < 0.0) { return outdoor_radiance(d); }
  let pr = prims[h.id];
  let x = p + d * h.t;
  let n = h.n;
  let l = frame.toSun;
  var e = vec3f(0.0);
  let ndl = dot(n, l);
  if (ndl > 0.0 && !room_occluded(x + n * 1e-3, l, 1e-4, 100.0)) {
    e = frame.sunColor.rgb * ndl;
  }
  if (bounce > 1u) {
    e += probe_irradiance(sample_probes(x, n, n), n);
  }
  return pr.color.rgb * e / PI + frame.lampColor.rgb * pr.color.a;
}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

/// A uniformly random rotation (Shoemake's quaternion) for this probe and sweep, so successive
/// sweeps sample new directions and their average converges.
fn ray_rotation(probe: u32, seed: u32) -> mat3x3f {
  let h1 = pcg(probe * 9781u + seed * 6271u + 1u);
  let h2 = pcg(h1);
  let h3 = pcg(h2);
  let u = vec3f(f32(h1), f32(h2), f32(h3)) / 4294967296.0;
  let a = sqrt(1.0 - u.x);
  let b = sqrt(u.x);
  let q = vec4f(a * sin(TAU * u.y), a * cos(TAU * u.y), b * sin(TAU * u.z), b * cos(TAU * u.z));
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3f(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
    vec3f(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
    vec3f(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)));
}

var<workgroup> wg_sh: array<array<vec3f, 9>, 64>;

/// One workgroup per probe: `bake.rays` rays projected onto SH, reduced in workgroup memory, and
/// blended into the volume with weight `bake.blend`.
@compute @workgroup_size(64)
fn cs_probe(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let probe = bake.offset + wg.x;
  if (probe >= frame.gridDims.w) { return; }
  let p = probe_pos(probe_coord(probe));
  // Probes buried in furniture see only its inside: switch them off.
  let alive = !room_inside_solid(p, 0.0);
  var acc: array<vec3f, 9>;
  if (alive) {
    let rot = ray_rotation(probe, bake.seed);
    for (var i = li; i < bake.rays; i += 64u) {
      let d = rot * ray_dir(i, bake.rays);
      let radiance = trace_radiance(p, d, bake.bounce);
      var y = sh9_basis(d);
      for (var k = 0u; k < 9u; k++) { acc[k] += radiance * y[k]; }
    }
  }
  wg_sh[li] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) {
      for (var k = 0u; k < 9u; k++) { wg_sh[li][k] += wg_sh[li + s][k]; }
    }
    workgroupBarrier();
  }
  if (li < 9u) {
    // Monte Carlo weight 4π/N, then the cosine lobe: stored coefficients give irradiance directly.
    var bands = SH_COSINE_BANDS;
    let c = wg_sh[0][li] * (4.0 * PI / f32(bake.rays)) * bands[li];
    let slot = probe * 9u + li;
    probes_out[slot] = vec4f(mix(probes_out[slot].rgb, c, bake.blend), select(0.0, select(0.0, 1.0, alive), li == 0u));
  }
}

var<workgroup> wg_dist: array<f32, VIS_RAYS>;
var<workgroup> wg_moments: array<vec2f, 64>;

/// One workgroup per probe: hit distances, filtered into an 8×8 octahedral map of distance
/// moments (one texel per thread), plus the border that makes bilinear lookups seamless.
@compute @workgroup_size(64)
fn cs_visibility(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let probe = bake.offset + wg.x;
  if (probe >= frame.gridDims.w) { return; }
  let c = probe_coord(probe);
  let p = probe_pos(c);
  for (var i = li; i < VIS_RAYS; i += 64u) {
    let h = room_trace(p, ray_dir(i, VIS_RAYS), 1e-4, MAX_DIST);
    wg_dist[i] = select(MAX_DIST, h.t, h.t >= 0.0);
  }
  workgroupBarrier();

  let tx = li % OCT;
  let ty = li / OCT;
  let dir = oct_decode(((vec2f(f32(tx), f32(ty)) + 0.5) / f32(OCT)) * 2.0 - 1.0);
  // A cosine-power lobe per texel, wide enough to average about ten rays.
  var m = vec2f(0.0);
  var wsum = 0.0;
  for (var i = 0u; i < VIS_RAYS; i++) {
    let w = pow(max(0.0, dot(dir, ray_dir(i, VIS_RAYS))), VIS_SHARPNESS);
    let d = wg_dist[i];
    m += w * vec2f(d, d * d);
    wsum += w;
  }
  m = select(vec2f(MAX_DIST, MAX_DIST * MAX_DIST), m / wsum, wsum > 1e-6);
  wg_moments[li] = m;
  let origin = probe_tile(c);
  textureStore(atlas_out, origin + 1u + vec2u(tx, ty), vec4f(m, 0.0, 1.0));
  workgroupBarrier();

  // Border: octahedral wrap mirrors each edge (and swaps opposite corners).
  if (li < 36u) {
    var dst: vec2i;
    var src: vec2u;
    let k = li % 8u;
    let last = OCT - 1u;
    switch li / 8u {
      case 0u: { dst = vec2i(i32(k), -1); src = vec2u(last - k, 0u); }
      case 1u: { dst = vec2i(i32(k), i32(OCT)); src = vec2u(last - k, last); }
      case 2u: { dst = vec2i(-1, i32(k)); src = vec2u(0u, last - k); }
      case 3u: { dst = vec2i(i32(OCT), i32(k)); src = vec2u(last, last - k); }
      default: {
        let corner = vec2u(k & 1u, k >> 1u);
        dst = select(vec2i(-1), vec2i(i32(OCT)), corner == vec2u(1u));
        src = select(vec2u(last), vec2u(0u), corner == vec2u(1u));
      }
    }
    let v = wg_moments[src.y * OCT + src.x];
    textureStore(atlas_out, vec2u(vec2i(origin) + 1 + dst), vec4f(v, 0.0, 1.0));
  }
}

// --- Light pass -----------------------------------------------------------------------

@vertex
fn vs_shadow(v: VertexIn, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return frame.lightViewProj * objects[id].model * vec4f(v.position, 1.0);
}

// --- Main pass ------------------------------------------------------------------------

struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) id: u32,
};

@vertex
fn vs_main(v: VertexIn, @builtin(instance_index) id: u32) -> VsOut {
  let m = objects[id].model;
  let wp = m * vec4f(v.position, 1.0);
  var out: VsOut;
  out.position = frame.viewProj * wp;
  out.world_pos = wp.xyz;
  // Boxes are scaled along their own axes, so m·n still points along the face normal.
  out.normal = normalize((m * vec4f(v.normal, 0.0)).xyz);
  out.id = id;
  return out;
}

fn sun_visibility(p: vec3f, n: vec3f, ndl: f32) -> f32 {
  let offset = n * frame.normalOffset * frame.shadowTexelWorld * (1.0 - ndl * 0.5);
  let clip = frame.lightViewProj * vec4f(p + offset, 1.0);
  let lp = vec3f(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5, clip.z - 0.0005);
  if (!shadow_in_bounds(lp)) { return 1.0; }
  let texel = 1.0 / f32(textureDimensions(shadow_map).x);
  return shadow_pcf_grid(lp, texel, 1) * smoothstep(0.0, 0.08, ndl);
}

const VIEW_INDIRECT = 1u;
const VIEW_IRRADIANCE = 2u;
const VIEW_DIRECT = 3u;
const VIEW_TRUST = 4u;

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4f {
  let obj = objects[v.id];
  let mat = u32(obj.material.y + 0.5);
  let vdir = normalize(frame.cameraPos - v.world_pos);
  if (mat == ROOM_MAT_SKY) { return vec4f(outdoor_radiance(-vdir), 1.0); }
  let n = normalize(v.normal);
  if (room_is_cut(v.world_pos)) { return vec4f(ROOM_CUT_COLOR * (0.7 + 0.3 * n.y), 1.0); }
  let albedo = room_albedo(mat, obj.color.rgb, v.world_pos, n);
  let m = Material(albedo, obj.color.a, obj.material.x);

  let l = frame.toSun;
  let ndl = saturate(dot(n, l));
  var direct = vec3f(0.0);
  if (ndl > 0.0) { direct = frame.sunColor.rgb * sun_visibility(v.world_pos, n, ndl) * brdf_direct(m, n, vdir, l); }

  // Bias toward the viewer as well as along the normal (DDGI): it keeps lookups on the visible
  // side of thin geometry.
  let lookup = sample_probes(v.world_pos, n, normalize(n * 0.2 + vdir * 0.8));
  let e = probe_irradiance(lookup, n) * frame.indirect;
  // The same SH, evaluated along the reflection, stands in for a blurry specular environment.
  let e_r = probe_irradiance(lookup, reflect(-vdir, n)) * frame.indirect;
  let indirect = brdf_ambient(m, n, vdir, e / PI, e_r / PI);
  let emissive = select(vec3f(0.0), frame.lampColor.rgb, mat == ROOM_MAT_LAMP);

  var col = direct + indirect + emissive;
  switch frame.debugView {
    case VIEW_INDIRECT: { col = indirect + emissive; }
    case VIEW_IRRADIANCE: { col = e / PI; }
    case VIEW_DIRECT: { col = direct + emissive; }
    case VIEW_TRUST: {
      let t = lookup.trust;
      col = mix(vec3f(0.9, 0.15, 0.1), vec3f(0.95, 0.9, 0.8), t) * (0.35 + 0.65 * saturate(dot(n, vec3f(0.3, 0.8, 0.5)) * 0.5 + 0.5));
    }
    default: {}
  }
  return vec4f(col, 1.0);
}

// --- Debug: probe spheres and the grid -------------------------------------------------

struct ProbeOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) @interpolate(flat) probe: u32,
};

@vertex
fn vs_probe(v: VertexIn, @builtin(instance_index) probe: u32) -> ProbeOut {
  let p = probe_pos(probe_coord(probe)) + v.position * frame.probeRadius;
  var out: ProbeOut;
  out.position = frame.viewProj * vec4f(p, 1.0);
  out.normal = v.normal;
  out.probe = probe;
  return out;
}

/// Each sphere shows its own probe's SH: irradiance over π, a white diffuse ball lit by it.
@fragment
fn fs_probe(v: ProbeOut) -> @location(0) vec4f {
  let idx = v.probe * 9u;
  if (probes[idx].w < 0.5) { return vec4f(0.35, 0.02, 0.02, 1.0); }
  var sh: array<vec3f, 9>;
  for (var k = 0u; k < 9u; k++) { sh[k] = probes[idx + k].rgb; }
  return vec4f(sh9_irradiance(sh, normalize(v.normal), frame.shOrder, frame.dering) / PI * frame.indirect, 1.0);
}

/// Grid lines: instance = probe × 3 + axis, a segment to the next probe along that axis.
@vertex
fn vs_grid(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> @builtin(position) vec4f {
  let c = probe_coord(inst / 3u);
  var axis = vec3u(0u);
  axis[inst % 3u] = 1u;
  let end = c + axis * vi;
  // Segments past the last probe collapse onto their start.
  let valid = all(c + axis < frame.gridDims.xyz);
  let p = probe_pos(select(c, end, valid));
  return frame.viewProj * vec4f(p, 1.0);
}

@fragment
fn fs_grid() -> @location(0) vec4f {
  return vec4f(0.22, 0.2, 0.17, 1.0);
}
