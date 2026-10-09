/**
 * Mip-chain generation for 2D colour textures via successive render passes
 * (each level is a bilinear 2×2 downsample of the previous one). Works for any
 * renderable, filterable format, including the *-srgb ones: filtering and
 * blending happen in linear space automatically.
 */

const pipelines = new WeakMap<GPUDevice, Map<GPUTextureFormat, GPURenderPipeline>>();
const samplers = new WeakMap<GPUDevice, GPUSampler>();

const MIP_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;

struct Out { @builtin(position) position: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return Out(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y));
}

@fragment fn fs(v: Out) -> @location(0) vec4f {
  return textureSampleLevel(src, smp, v.uv, 0.0);
}`;

export function mipLevelCount(width: number, height: number): number {
  return Math.floor(Math.log2(Math.max(width, height))) + 1;
}

function pipelineFor(device: GPUDevice, format: GPUTextureFormat): GPURenderPipeline {
  let byFormat = pipelines.get(device);
  if (!byFormat) pipelines.set(device, (byFormat = new Map()));
  let p = byFormat.get(format);
  if (!p) {
    const module = device.createShaderModule({ label: 'mipmaps', code: MIP_WGSL });
    p = device.createRenderPipeline({
      label: `mipmaps:${format}`,
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    });
    byFormat.set(format, p);
  }
  return p;
}

/** Fills mips 1..n-1 of `texture` (needs RENDER_ATTACHMENT | TEXTURE_BINDING). */
export function generateMipmaps(device: GPUDevice, texture: GPUTexture, encoder?: GPUCommandEncoder): void {
  if (texture.mipLevelCount < 2) return;
  const pipeline = pipelineFor(device, texture.format);
  let sampler = samplers.get(device);
  if (!sampler) samplers.set(device, (sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' })));
  const enc = encoder ?? device.createCommandEncoder({ label: 'mipmaps' });
  for (let level = 1; level < texture.mipLevelCount; level++) {
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) },
        { binding: 1, resource: sampler },
      ],
    });
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view: texture.createView({ baseMipLevel: level, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
  }
  if (!encoder) device.queue.submit([enc.finish()]);
}
