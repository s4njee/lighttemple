// ---------------------------------------------------------------------------
// jfa.wgsl — signed distance field from an anti-aliased coverage mask via the
// Jump Flooding Algorithm (Rong & Tan 2006).
//
//   cs_seed:  pixels on either side of the 0.5-coverage boundary become seeds.
//             Each stores the *sub-pixel* edge point, estimated from the
//             coverage value and its gradient, so the field is accurate to a
//             fraction of a pixel (plain JFA is only accurate to ~0.5 px).
//   cs_flood: log2(N) passes with halving step: adopt a neighbour's seed if closer.
//   cs_resolve: signed Euclidean distance in pixels (negative inside) -> rgba16float.
// ---------------------------------------------------------------------------

struct JfaParams { step: i32 };

@group(0) @binding(0) var mask: texture_2d<f32>;
@group(0) @binding(1) var seeds_in: texture_2d<f32>;
@group(0) @binding(2) var seeds_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var sdf_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> jfa: JfaParams;

const NO_SEED = vec2f(-1e6, -1e6);

fn coverage(p: vec2i) -> f32 {
  let dims = vec2i(textureDimensions(mask));
  if (any(p < vec2i(0)) || any(p >= dims)) { return 0.0; }
  return textureLoad(mask, p, 0).a;
}

@compute @workgroup_size(8, 8)
fn cs_seed(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(mask));
  let p = vec2i(id.xy);
  if (any(p >= dims)) { return; }
  let c = coverage(p);
  let inside = c >= 0.5;
  let boundary =
    (coverage(p + vec2i(1, 0)) >= 0.5) != inside || (coverage(p - vec2i(1, 0)) >= 0.5) != inside ||
    (coverage(p + vec2i(0, 1)) >= 0.5) != inside || (coverage(p - vec2i(0, 1)) >= 0.5) != inside;
  var seed = NO_SEED;
  if (boundary) {
    let g = vec2f(coverage(p + vec2i(1, 0)) - coverage(p - vec2i(1, 0)), coverage(p + vec2i(0, 1)) - coverage(p - vec2i(0, 1))) * 0.5;
    let center = vec2f(p) + 0.5;
    let gl = length(g);
    // Coverage changes by |g| per pixel along g; walk to where it equals 0.5.
    let offset = select(vec2f(0.0), -g / gl * clamp((c - 0.5) / gl, -1.0, 1.0), gl > 1e-4);
    seed = center + offset;
  }
  textureStore(seeds_out, p, vec4f(seed, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn cs_flood(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(seeds_in));
  let p = vec2i(id.xy);
  if (any(p >= dims)) { return; }
  let here = vec2f(p) + 0.5;
  var best = textureLoad(seeds_in, p, 0).xy;
  var best_d = select(1e12, dot(best - here, best - here), best.x > -1e5);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = p + vec2i(x, y) * jfa.step;
      if (any(q < vec2i(0)) || any(q >= dims)) { continue; }
      let s = textureLoad(seeds_in, q, 0).xy;
      if (s.x <= -1e5) { continue; }
      let d = dot(s - here, s - here);
      if (d < best_d) {
        best_d = d;
        best = s;
      }
    }
  }
  textureStore(seeds_out, p, vec4f(best, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn cs_resolve(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(seeds_in));
  let p = vec2i(id.xy);
  if (any(p >= dims)) { return; }
  let s = textureLoad(seeds_in, p, 0).xy;
  let here = vec2f(p) + 0.5;
  let dist = select(1e4, length(s - here), s.x > -1e5);
  let d = select(dist, -dist, coverage(p) >= 0.5);
  textureStore(sdf_out, p, vec4f(d, 0.0, 0.0, 1.0));
}
