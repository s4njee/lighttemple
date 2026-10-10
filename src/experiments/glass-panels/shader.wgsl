// ---------------------------------------------------------------------------
// Glass Panels — frosted "liquid glass" UI cards over a lit background.
//
//   fs_background: animated light blobs, a big headline and a faint grid, with
//                  the panels' soft drop shadows (cast away from the light).
//                  Rendered to an HDR texture, then mip-mapped for blur.
//   fs_glass:      for each pixel inside a panel (rounded-rect SDF):
//                  - a circular bevel profile gives a surface normal near the edge,
//                  - refraction offsets the background lookup along that normal
//                    (flat centre = no bend, edges bend strongly),
//                  - frost = jittered taps at a blurrier mip,
//                  - dispersion = separate R/G/B offsets,
//                  - specular rim from the pointer light + Fresnel edge glow,
//                  - a 1 px hairline and simple "UI" content on top.
// Requires (prepended): common.wgsl, sdf.wgsl, fullscreen.wgsl, `Glass`.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> g: Glass;
@group(0) @binding(1) var text_sdf: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var bg_tex: texture_2d<f32>;

fn panel(i: i32) -> vec4f {
  switch i {
    case 0: { return g.panel0; }
    case 1: { return g.panel1; }
    default: { return g.panel2; }
  }
}

fn panel_radius(i: i32) -> f32 { return g.radii[i]; }

fn sd_panel(p: vec2f, i: i32) -> f32 {
  let r = panel(i);
  return sd_round_box(p - r.xy, r.zw, panel_radius(i));
}

fn sd_headline(p: vec2f) -> f32 {
  let rel = p - g.textRect.xy;
  let half = g.textRect.zw;
  let box = sd_box(rel, half);
  let uv = clamp(rel / half * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
  let d = textureSampleLevel(text_sdf, lin, uv, 0.0).r * g.textScale;
  return select(d, max(box, d), box > 0.0);
}

// --- Background (pixel space, y down) ---------------------------------------------

fn blob(p: vec2f, c: vec2f, r: f32, col: vec3f) -> vec3f {
  let d = length(p - c) / r;
  return col * exp(-d * d * 2.0);
}

@fragment
fn fs_background(v: FullscreenOut) -> @location(0) vec4f {
  let p = v.position.xy;
  let res = g.resolution;
  let t = g.time;
  let s = res.y;
  var c = mix(vec3f(0.010, 0.012, 0.030), vec3f(0.03, 0.02, 0.05), v.uv.y);
  c += blob(p, res * vec2f(0.28 + 0.08 * sin(t * 0.21), 0.35 + 0.1 * sin(t * 0.17)), 0.36 * s, vec3f(1.1, 0.36, 0.08));
  c += blob(p, res * vec2f(0.72 + 0.07 * sin(t * 0.19 + 2.0), 0.3 + 0.08 * sin(t * 0.23 + 1.0)), 0.32 * s, vec3f(0.65, 0.08, 0.55));
  c += blob(p, res * vec2f(0.55 + 0.1 * sin(t * 0.13 + 4.0), 0.78 + 0.06 * sin(t * 0.29)), 0.38 * s, vec3f(0.05, 0.35, 0.95));
  c += blob(p, res * vec2f(0.12 + 0.05 * sin(t * 0.31 + 3.0), 0.85), 0.22 * s, vec3f(0.15, 0.6, 0.38));

  // Faint grid and a big outlined headline: detail for the glass to bend.
  let grid = 0.5 * s / 10.0;
  let gl = min(abs(fract(p.x / grid) - 0.5), abs(fract(p.y / grid) - 0.5)) * grid;
  c += vec3f(0.035) * (1.0 - smoothstep(0.0, 1.2, gl));
  let td = sd_headline(p);
  c = mix(c, c * 0.25 + vec3f(0.85, 0.82, 0.8), smoothstep(1.0, -1.0, td));

  // Soft drop shadows, cast away from the light.
  var shade = 1.0;
  for (var i = 0; i < 3; i++) {
    let r = panel(i);
    let away = normalize(r.xy - g.light.xy + vec2f(1e-3, 0.0));
    let offset = away * g.shadowOffset + vec2f(0.0, g.shadowOffset * 0.5);
    let d = sd_round_box(p - r.xy - offset, r.zw, panel_radius(i));
    shade *= mix(1.0 - g.shadowStrength, 1.0, smoothstep(-g.shadowBlur, g.shadowBlur, d));
  }
  return vec4f(c * shade, 1.0);
}

// --- Glass ------------------------------------------------------------------------------

fn bg(uv: vec2f, lod: f32) -> vec3f {
  return textureSampleLevel(bg_tex, lin, uv, lod).rgb;
}

/// Frosted lookup: 8 rotated taps at a blurrier mip (smoother than one mip tap).
fn frosted(uv: vec2f, lod: f32, angle: f32) -> vec3f {
  if (lod < 0.05) { return bg(uv, 0.0); }
  let radius = exp2(lod) / g.resolution * 0.9;
  var sum = bg(uv, max(lod - 1.0, 0.0));
  for (var k = 0; k < 7; k++) {
    let a = angle + f32(k) * 0.8975979; // 2π/7
    let r = sqrt((f32(k) + 0.5) / 7.0);
    sum += bg(uv + vec2f(cos(a), sin(a)) * radius * r, max(lod - 1.0, 0.0));
  }
  return sum / 8.0;
}

/// Skeleton UI drawn on the main panel: a title bar, an avatar and text lines.
fn ui_content(p: vec2f, r: vec4f) -> f32 {
  let q = p - (r.xy - r.zw); // top-left origin
  let w = 2.0 * r.z;
  let h = 2.0 * r.w;
  var d = sd_circle(q - vec2f(0.12 * w, 0.2 * h), 0.07 * h);
  d = min(d, sd_round_box(q - vec2f(0.5 * w, 0.17 * h), vec2f(0.24 * w, 0.025 * h), 0.025 * h));
  d = min(d, sd_round_box(q - vec2f(0.44 * w, 0.25 * h), vec2f(0.18 * w, 0.018 * h), 0.018 * h));
  for (var i = 0; i < 4; i++) {
    let y = (0.45 + 0.11 * f32(i)) * h;
    let len = select(0.38, 0.27, i == 3) * w;
    d = min(d, sd_round_box(q - vec2f(0.1 * w + len, y), vec2f(len, 0.018 * h), 0.018 * h));
  }
  return d;
}

@fragment
fn fs_glass(v: FullscreenOut) -> @location(0) vec4f {
  let p = v.position.xy;
  let uv = p / g.resolution;
  var col = bg(uv, 0.0);

  // Draw panels back to front (2 = back, 0 = front).
  for (var i = 2; i >= 0; i--) {
    let d = sd_panel(p, i);
    let coverage = smoothstep(0.75, -0.75, d);
    if (coverage <= 0.0) { continue; }

    // Bevel: circular profile of width `bevel` px; normal from the SDF gradient.
    let e = 1.0;
    let grad = normalize(vec2f(sd_panel(p + vec2f(e, 0.0), i) - sd_panel(p - vec2f(e, 0.0), i),
                               sd_panel(p + vec2f(0.0, e), i) - sd_panel(p - vec2f(0.0, e), i)) + vec2f(1e-6, 0.0));
    let x = saturate(1.0 + d / g.bevel); // 1 at the edge, 0 where the flat top begins
    let slope = x / sqrt(max(1.0 - x * x, 1e-3)); // derivative of a circular arc
    let n = normalize(vec3f(grad * min(slope, 6.0), 1.0));

    // Refraction (screen-space approximation): bend along the surface tilt.
    let bend = n.xy * g.refraction;
    let angle = ign(p) * TAU;
    let off = bend / g.resolution;
    let k = g.dispersion;
    var glass = vec3f(
      frosted(uv - off * (1.0 - k), g.frost, angle).r,
      frosted(uv - off, g.frost, angle).g,
      frosted(uv - off * (1.0 + k), g.frost, angle).b);
    // Frosted glass scatters a bit of light: lift and tint.
    glass = mix(glass, glass * g.tint.rgb + g.tint.a * 0.06, 0.6);

    // Pointer light: specular glints on the bevel, Fresnel rim, hairline edge.
    let lpos = vec3f(g.light.xy, g.light.z);
    let l = normalize(lpos - vec3f(p, 0.0));
    let h = normalize(l + vec3f(0.0, 0.0, 1.0));
    let spec = pow(saturate(dot(n, h)), 90.0) * 3.0 + pow(saturate(dot(n, h)), 12.0) * 0.15;
    let fresnel = pow(1.0 - saturate(n.z), 3.0);
    let lit = saturate(dot(n.xy, normalize(g.light.xy - p + vec2f(1e-3, 0.0)))) * 0.5 + 0.5;
    glass += g.lightColor * spec + vec3f(0.35, 0.38, 0.45) * fresnel * lit * g.rim;
    let hairline = smoothstep(-2.0, -0.5, d) * smoothstep(0.5, -0.5, d);
    glass += vec3f(0.5) * hairline * (0.3 + 0.7 * lit) * g.rim;

    // Content on the front (main) panel.
    if (i == 0) {
      let ud = ui_content(p, panel(0));
      glass = mix(glass, vec3f(0.95, 0.96, 1.0) * 0.85, smoothstep(0.75, -0.75, ud) * 0.75);
    }
    col = mix(col, glass, coverage);
  }
  return vec4f(col, 1.0);
}
