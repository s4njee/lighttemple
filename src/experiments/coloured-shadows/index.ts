import commonWgsl from '../../shaders/common.wgsl?raw';
import fullscreenWgsl from '../../shaders/fullscreen.wgsl?raw';
import { createShader, wgsl } from '../../gpu/shader';
import { UniformBlock } from '../../gpu/uniforms';
import { HDR_FORMAT, PostProcess } from '../../post/post';
import { hexToLinear } from '../../gpu/color';
import type { ExperimentFactory } from '../types';
import shaderWgsl from './shader.wgsl?raw';

const create: ExperimentFactory = async (ctx) => {
  const { device, surface, params, pointer } = ctx;

  // Param keys must be unique across the panel (they are also the URL keys).
  const light = params.add(
    {
      intensity: { value: 1.5, min: 0, max: 6, step: 0.01 },
      color: { value: '#ffd2a1', type: 'color' },
      height: { value: 0.3, min: 0.02, max: 1.5, step: 0.01 },
    },
    'Light',
  );
  const post = await PostProcess.create(ctx);

  // Declared once here; `uniforms.wgslStruct` is spliced into the shader.
  const uniforms = new UniformBlock(device, 'Params', {
    resolution: 'vec2f',
    time: 'f32',
    intensity: 'f32',
    light: 'vec4f',
    lightColor: 'vec4f',
  });

  const module = await createShader(device, 'coloured-shadows', wgsl(commonWgsl, fullscreenWgsl, uniforms.wgslStruct, shaderWgsl));
  const pipeline = await device.createRenderPipelineAsync({
    label: 'coloured-shadows',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_fullscreen' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
  });

  return {
    frame({ time, size }) {
      const aspect = size.width / size.height;
      const [u, v] = pointer.smoothUv;
      uniforms
        .set('resolution', [size.width, size.height])
        .set('time', time)
        .set('intensity', light.intensity)
        .set('light', [(u - 0.5) * 2 * aspect, (0.5 - v) * 2, light.height, 0])
        .set('lightColor', [...hexToLinear(light.color), 0])
        .upload();

      const encoder = device.createCommandEncoder({ label: 'coloured-shadows' });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: post.scene(size).view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      post.render(encoder, surface.currentTexture().createView(), { time });
      device.queue.submit([encoder.finish()]);
    },
    dispose() {
      post.destroy();
      uniforms.destroy();
    },
  };
};

export default create;
