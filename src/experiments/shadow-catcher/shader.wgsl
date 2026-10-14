// ---------------------------------------------------------------------------
// Shadow Catcher — a 3D object floating over live HTML.
//
// World space: the page is the plane z = 0 (x right, y up, +z toward the
// viewer); the camera looks straight down -z at it, so the plane lines up with
// the HTML behind the transparent canvas.
//
//   object:  lit with the shared PBR model (key light + a hemisphere between
//            "room" above and the page's own colour bouncing up from below),
//            self-shadowed from the key map with a small PCF kernel.
//   page:    a fullscreen shadow catcher on z = 0 that writes premultiplied
//            black with alpha = darkness, so it only ever darkens the HTML:
//            - a PCSS key shadow (blocker search, then a filter sized by the
//              receiver–blocker distance): crisp where the object nearly
//              touches the page, soft when it hovers high;
//            - contact occlusion from a depth map rendered looking *up* from
//              below the page: each texel holds how high the object's
//              underside is above that point.
//   inspect: debug views that read back the post chain's final output.
//
// Requires (prepended): common.wgsl, fullscreen.wgsl, VertexIn, Object (props),
// brdf.wgsl, the `Frame` struct.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var key_map: texture_depth_2d;
@group(0) @binding(3) var contact_map: texture_depth_2d;
@group(0) @binding(4) var shadow_cmp: sampler_comparison;

const GOLDEN_ANGLE: f32 = 2.39996323;

fn to_shadow_uv(m: mat4x4f, p: vec3f) -> vec3f {
  let c = m * vec4f(p, 1.0);
  let ndc = c.xyz / c.w;
  return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

/// Point `i` of an `n`-point Vogel (sunflower) disc of radius 1, rotated by `angle`.
fn vogel(i: u32, n: u32, angle: f32) -> vec2f {
  let r = sqrt((f32(i) + 0.5) / f32(n));
  let t = f32(i) * GOLDEN_ANGLE + angle;
  return r * vec2f(cos(t), sin(t));
}

// --- Depth-only passes ---------------------------------------------------------

@vertex
fn vs_key(v: VertexIn, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return frame.keyViewProj * objects[id].model * vec4f(v.position, 1.0);
}

@vertex
fn vs_contact(v: VertexIn, @builtin(instance_index) id: u32) -> @builtin(position) vec4f {
  return frame.contactViewProj * objects[id].model * vec4f(v.position, 1.0);
}

// --- Object ------------------------------------------------------------------------

struct ObjectOut {
  @builtin(position) position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) id: u32,
};

@vertex
fn vs_object(v: VertexIn, @builtin(instance_index) id: u32) -> ObjectOut {
  let m = objects[id].model;
  let wp = m * vec4f(v.position, 1.0);
  // Props may be non-uniformly scaled (pebbles): use the inverse-transpose for normals.
  let m3 = mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
  let nm = transpose(inverse3(m3));
  var out: ObjectOut;
  out.position = frame.viewProj * wp;
  out.world_pos = wp.xyz;
  out.normal = nm * v.normal;
  out.id = id;
  return out;
}

fn inverse3(m: mat3x3f) -> mat3x3f {
  let a = m[0]; let b = m[1]; let c = m[2];
  let r0 = cross(b, c); let r1 = cross(c, a); let r2 = cross(a, b);
  let inv_det = 1.0 / dot(a, r0);
  return transpose(mat3x3f(r0, r1, r2)) * inv_det;
}

@fragment
fn fs_object(v: ObjectOut) -> @location(0) vec4f {
  let obj = objects[v.id];
  let n = normalize(v.normal);
  let l = frame.toLight;
  let vdir = normalize(frame.cameraPos - v.world_pos);
  let ndl = saturate(dot(n, l));

  // Self-shadowing: normal-offset lookup + 3×3 bilinear PCF.
  let texel = 1.0 / f32(textureDimensions(key_map).x);
  let offset = n * frame.keyTexelWorld * 1.5 * (1.0 - 0.5 * ndl);
  let lp = to_shadow_uv(frame.keyViewProj, v.world_pos + offset);
  var vis = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      vis += textureSampleCompareLevel(key_map, shadow_cmp, lp.xy + vec2f(f32(x), f32(y)) * texel, lp.z - 0.002);
    }
  }
  vis = vis / 9.0 * smoothstep(0.0, 0.1, ndl);

  let m = Material(obj.color.rgb, obj.color.a, obj.material.x);
  // Hemisphere "room": soft light from in front of the page, the page's own colour from behind.
  let irradiance = mix(frame.bounceColor, frame.skyColor, n.z * 0.5 + 0.5);
  let r = reflect(-vdir, n);
  let edge = mix(0.05, 0.8, m.roughness);
  let env = mix(frame.bounceColor, frame.skyColor, smoothstep(-edge, edge, r.z));
  let col = brdf_ambient(m, n, vdir, irradiance, env) + frame.lightColor * vis * brdf_direct(m, n, vdir, l);
  if (frame.debugView == 3u) { return vec4f(vec3f(0.35) * (0.4 + 0.6 * vis), 1.0); }
  return vec4f(col, 1.0);
}

// --- Page shadow catcher ---------------------------------------------------------------

/// PCSS for a directional light of angular radius atan(lightTan): returns lit fraction.
fn key_visibility(lp: vec3f, frag: vec2f) -> f32 {
  if (any(lp.xy <= vec2f(0.0)) || any(lp.xy >= vec2f(1.0))) { return 1.0; }
  let dims = vec2f(textureDimensions(key_map));
  let texel = 1.0 / dims.x;
  let world_to_uv = 1.0 / (2.0 * frame.keyExtent);
  let angle = ign(frag) * TAU;
  let n = frame.taps;

  // 1. Blocker search over the region that could hold occluders for this receiver.
  let search_uv = clamp(frame.lightTan * lp.z * frame.keyDepthRange * world_to_uv, 2.0 * texel, 0.25);
  var blocker_sum = 0.0;
  var blockers = 0.0;
  let search_taps = max(n / 2u, 4u);
  for (var i = 0u; i < search_taps; i++) {
    let uv = lp.xy + vogel(i, search_taps, angle) * search_uv;
    let px = clamp(vec2i(uv * dims), vec2i(0), vec2i(dims) - 1);
    let d = textureLoad(key_map, px, 0);
    if (d < lp.z) {
      blocker_sum += d;
      blockers += 1.0;
    }
  }
  if (blockers == 0.0) { return 1.0; }
  let avg_blocker = blocker_sum / blockers;

  // 2. Penumbra width: proportional to how far the receiver is behind the blocker.
  let penumbra_uv = frame.lightTan * (lp.z - avg_blocker) * frame.keyDepthRange * world_to_uv;
  let radius = max(penumbra_uv, 1.5 * texel);

  // 3. Filter: Vogel-disc PCF with that radius, each tap a bilinear comparison.
  var lit = 0.0;
  for (var i = 0u; i < n; i++) {
    lit += textureSampleCompareLevel(key_map, shadow_cmp, lp.xy + vogel(i, n, angle) * radius, lp.z);
  }
  return lit / f32(n);
}

/// How occluded this bit of page is by the object's underside directly above it.
fn contact_occlusion(p: vec3f, frag: vec2f) -> f32 {
  let cp = to_shadow_uv(frame.contactViewProj, p);
  if (any(cp.xy <= vec2f(0.0)) || any(cp.xy >= vec2f(1.0))) { return 0.0; }
  let dims = vec2f(textureDimensions(contact_map));
  let radius = frame.contactRadius / (2.0 * frame.contactExtent);
  let angle = ign(frag + 17.0) * TAU;
  var occ = 0.0;
  for (var i = 0u; i < 16u; i++) {
    let uv = cp.xy + vogel(i, 16u, angle) * radius;
    let px = clamp(vec2i(uv * dims), vec2i(0), vec2i(dims) - 1);
    let d = textureLoad(contact_map, px, 0);
    occ += select(0.0, exp(-d * frame.contactRange / max(frame.contactFalloff, 1e-3)), d < 1.0);
  }
  return occ / 16.0;
}

@fragment
fn fs_page(v: FullscreenOut) -> @location(0) vec4f {
  // Intersect the pixel's view ray with the page plane z = 0.
  let ndc = vec2f(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0);
  let a = frame.invViewProj * vec4f(ndc, 0.0, 1.0);
  let b = frame.invViewProj * vec4f(ndc, 1.0, 1.0);
  let o = a.xyz / a.w;
  let d = b.xyz / b.w - o;
  let p = o - d * (o.z / d.z);

  let key_dark = (1.0 - key_visibility(to_shadow_uv(frame.keyViewProj, p), v.position.xy)) * frame.shadowOpacity;
  let contact_dark = contact_occlusion(p, v.position.xy) * frame.contactStrength;
  if (frame.debugView == 3u) { return vec4f(key_dark, contact_dark, 0.0, 1.0); }
  let darkness = 1.0 - (1.0 - key_dark) * (1.0 - contact_dark);
  // Premultiplied black with alpha = darkness: only ever darkens what's behind the canvas.
  return vec4f(0.0, 0.0, 0.0, saturate(darkness));
}

// --- Inspect the final (post-chain) output -------------------------------------------

@group(0) @binding(5) var final_image: texture_2d<f32>;

@fragment
fn fs_inspect(v: FullscreenOut) -> @location(0) vec4f {
  let c = textureLoad(final_image, vec2i(v.position.xy), 0);
  if (frame.debugView == 1u) { return vec4f(vec3f(c.a), 1.0); }
  // Composite over a checkerboard exactly as the browser would: rgb + (1 - a) * page, in sRGB.
  let cell = (i32(floor(v.position.x / 16.0)) + i32(floor(v.position.y / 16.0))) & 1;
  let page = select(0.58, 0.86, cell == 1);
  // Invalid premultiplied pixels (colour brighter than alpha allows) show up red.
  if (any(c.rgb > vec3f(c.a) + 1e-4)) { return vec4f(1.0, 0.0, 0.0, 1.0); }
  return vec4f(c.rgb + (1.0 - c.a) * page, 1.0);
}
