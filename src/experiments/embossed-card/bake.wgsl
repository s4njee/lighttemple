// ---------------------------------------------------------------------------
// Embossed Card — height-field bake (runs only when the card design changes).
//
// Output (rgba16float), in card texture space:
//   r = height 0..1 (0.5 = card surface; emboss rises toward 1, deboss sinks toward 0)
//   g = design coverage 0..1 (where foil / ink goes)
//   b = roughness variation 0..1 (0.5 = none)
// Requires (prepended): common.wgsl, noise.wgsl, sdf.wgsl, fullscreen.wgsl, `Bake`.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> bake: Bake;
@group(0) @binding(1) var head_sdf: texture_2d<f32>;
@group(0) @binding(2) var sub_sdf: texture_2d<f32>;
@group(0) @binding(3) var lin: sampler;

/// Distance to text whose SDF texture is placed at rect = (centre, half-size) in card units.
fn sd_text_tex(tex: texture_2d<f32>, rect: vec4f, scale_pad: vec2f, p: vec2f) -> f32 {
  let rel = p - rect.xy;
  let half = rect.zw;
  let box = sd_box(rel, half);
  let uv = clamp(vec2f(rel.x / half.x, -rel.y / half.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
  let d = textureSampleLevel(tex, lin, uv, 0.0).r * scale_pad.x;
  let glyph_box = sd_box(rel, max(half - vec2f(scale_pad.y), vec2f(1e-3)));
  return select(d, max(glyph_box, d - box), box > 0.0);
}

/// A small "temple" mark: a ring around a triangle.
fn sd_logo(p: vec2f) -> f32 {
  let ring = sd_ring(p, 0.075, 0.006);
  let tri = sd_equilateral_triangle(p + vec2f(0.0, 0.006), 0.038);
  return min(ring, op_onion(tri, 0.005));
}

/// Material microstructure in ~[-1, 1] (x) and roughness variation (y).
fn micro(p: vec2f) -> vec2f {
  let q = p + vec2f(bake.seed * 13.7, bake.seed * 7.3);
  switch bake.material {
    // Cotton paper: soft cloudy formation + fine fibres in random directions.
    case 0u: {
      let cloud = fbm(q * 18.0, 4);
      let fibres = gradient_noise(rot2(0.6) * q * vec2f(260.0, 22.0)) + gradient_noise(rot2(-0.9) * q * vec2f(240.0, 25.0));
      return vec2f(cloud * 0.5 + fibres * 0.35, 0.5 + 0.2 * cloud);
    }
    // Brushed metal: long scratches along x, plus faint large-scale waviness.
    case 1u: {
      let brush = gradient_noise(vec2f(q.x * 6.0, q.y * 900.0)) * 0.7 + gradient_noise(vec2f(q.x * 2.0, q.y * 2400.0)) * 0.3;
      return vec2f(brush * 0.6 + fbm(q * 6.0, 3) * 0.3, 0.5 + 0.35 * brush);
    }
    // Pebbled leather: cellular grain with wrinkles; tops are slightly glossier.
    case 2u: {
      let c = worley(q * 70.0 + fbm(q * 9.0, 3) * 0.6);
      let pebble = smoothstep(0.0, 0.35, c.y - c.x) * 2.0 - 1.0;
      let wrinkle = fbm(q * 12.0, 4);
      return vec2f(pebble * 0.8 + wrinkle * 0.4, 0.5 - 0.3 * pebble);
    }
    // Smooth matte stock: almost flat.
    default: {
      return vec2f(fbm(q * 90.0, 3) * 0.4, 0.5);
    }
  }
}

@fragment
fn fs_bake(v: FullscreenOut) -> @location(0) vec4f {
  // Card units, origin at the centre, y up.
  let p = (v.uv - 0.5) * 2.0 * bake.cardHalf * vec2f(1.0, -1.0);

  var d = 1e9;
  d = min(d, sd_text_tex(head_sdf, bake.headRect, bake.headScalePad, p));
  d = min(d, sd_text_tex(sub_sdf, bake.subRect, bake.subScalePad, p));
  d = min(d, sd_logo(p - bake.logoPos));
  if (bake.border != 0u) {
    let inset = bake.cardHalf - vec2f(0.075);
    d = min(d, op_onion(sd_round_box(p, inset, 0.03), 0.0035));
  }

  // Pressed profile: a smooth shoulder of width `bevel`, like a real die.
  let profile = smoothstep(0.0, max(bake.bevel, 1e-4), -d);
  let m = micro(p);
  // Pressed areas are flattened by the die, so they keep less of the grain.
  let h = 0.5 + 0.45 * bake.relief * profile + bake.microAmount * m.x * (1.0 - 0.7 * profile);
  // Ink/foil covers the whole glyph (not just the fully pressed part), with ~1 texel of AA.
  let coverage = smoothstep(-0.0006, 0.0006, -d);
  return vec4f(clamp(h, 0.0, 1.0), coverage, m.y, 1.0);
}
