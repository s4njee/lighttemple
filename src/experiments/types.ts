/**
 * The contract between the gallery host and an experiment.
 *
 * The host owns the device, canvas, render loop, pointer and params panel.
 * An experiment owns its GPU resources and encodes its own passes each frame.
 */

import type { GpuContext } from '../gpu/context';
import type { Surface, SurfaceSize } from '../gpu/surface';
import type { ParamSink } from '../ui/params';
import type { PointerState } from '../ui/pointer';

export interface ExperimentContext {
  gpu: GpuContext;
  device: GPUDevice;
  surface: Surface;
  /** Swapchain format; render final output to `surface.currentTexture()`. */
  format: GPUTextureFormat;
  params: ParamSink;
  pointer: Readonly<PointerState>;
  /** The user prefers reduced motion; keep ambient animation subtle or static. */
  reducedMotion: boolean;
}

export interface FrameInfo {
  /** Seconds since the experiment started (pauses while the tab is hidden). */
  time: number;
  /** Seconds since the previous frame, clamped to avoid huge steps after a stall. */
  dt: number;
  frame: number;
  size: SurfaceSize;
}

export interface Experiment {
  /** Encode and submit one frame. */
  frame(info: FrameInfo): void;
  /** Called after the canvas backing store changes size (recreate size-dependent targets). */
  resize?(size: SurfaceSize): void;
  /** Release every GPU resource the experiment created. */
  dispose(): void;
}

export type ExperimentFactory = (ctx: ExperimentContext) => Promise<Experiment> | Experiment;

export type Epic =
  | 'ui-lighting'
  | 'shadows'
  | 'global-illumination'
  | 'volumetrics'
  | 'stylized'
  | 'many-lights'
  | 'web-integration';

export interface ExperimentMeta {
  /** URL slug: #/x/<slug> */
  slug: string;
  title: string;
  /** One-line pitch shown on the gallery card. */
  summary: string;
  epic: Epic;
  tags: string[];
  /** Longer "how it works" notes shown next to the canvas (plain text paragraphs). */
  notes: string[];
  /** Short CSS background for the gallery card until real thumbnails exist. */
  cardBackground: string;
  status: 'prototype' | 'stable';
  load: () => Promise<{ default: ExperimentFactory }>;
}
