/**
 * Damped orbit camera: drag to orbit, wheel/pinch to dolly.
 * Produces WebGPU-convention matrices (clip z in [0,1]) via wgpu-matrix.
 */

import { mat4, vec3, type Mat4, type Vec3 } from 'wgpu-matrix';

export interface OrbitCameraOptions {
  target?: [number, number, number];
  distance?: number;
  yaw?: number;
  pitch?: number;
  fovY?: number;
  near?: number;
  far?: number;
  minDistance?: number;
  maxDistance?: number;
}

export class OrbitCamera {
  target: Vec3;
  yaw: number;
  pitch: number;
  distance: number;
  fovY: number;
  near: number;
  far: number;
  minDistance: number;
  maxDistance: number;
  /** Radians/sec of idle auto-rotation; 0 to disable. */
  autoRotate = 0;

  readonly view: Mat4 = mat4.identity();
  readonly projection: Mat4 = mat4.identity();
  readonly viewProjection: Mat4 = mat4.identity();
  readonly position: Vec3 = vec3.create();

  private goal: { yaw: number; pitch: number; distance: number };
  private dragging: { id: number; x: number; y: number } | null = null;
  private pointers = new Map<number, { x: number; y: number }>();
  private pinchDist = 0;
  private disposers: (() => void)[] = [];

  constructor(
    private readonly element: HTMLElement,
    o: OrbitCameraOptions = {},
  ) {
    this.target = vec3.fromValues(...(o.target ?? [0, 0, 0]));
    this.yaw = o.yaw ?? 0.6;
    this.pitch = o.pitch ?? 0.45;
    this.distance = o.distance ?? 6;
    this.fovY = o.fovY ?? (40 * Math.PI) / 180;
    this.near = o.near ?? 0.1;
    this.far = o.far ?? 100;
    this.minDistance = o.minDistance ?? 1;
    this.maxDistance = o.maxDistance ?? 30;
    this.goal = { yaw: this.yaw, pitch: this.pitch, distance: this.distance };
    this.attach();
  }

  private attach(): void {
    const el = this.element;
    el.style.touchAction = 'none';
    const listen = <K extends keyof HTMLElementEventMap>(
      type: K,
      fn: (e: HTMLElementEventMap[K]) => void,
      opts?: AddEventListenerOptions,
    ) => {
      el.addEventListener(type, fn, opts);
      this.disposers.push(() => el.removeEventListener(type, fn));
    };
    listen('pointerdown', (e) => {
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.pointers.size === 1) this.dragging = { id: e.pointerId, x: e.clientX, y: e.clientY };
      else this.dragging = null;
      el.setPointerCapture(e.pointerId);
    });
    listen('pointermove', (e) => {
      if (!this.pointers.has(e.pointerId)) return;
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (this.pinchDist > 0) this.dolly(this.pinchDist / d);
        this.pinchDist = d;
        return;
      }
      if (!this.dragging || this.dragging.id !== e.pointerId) return;
      const dx = e.clientX - this.dragging.x;
      const dy = e.clientY - this.dragging.y;
      this.dragging.x = e.clientX;
      this.dragging.y = e.clientY;
      const s = 4 / Math.max(200, el.clientHeight);
      this.goal.yaw -= dx * s;
      this.goal.pitch = Math.min(1.45, Math.max(-0.2, this.goal.pitch + dy * s));
    });
    const end = (e: PointerEvent) => {
      this.pointers.delete(e.pointerId);
      if (this.pointers.size < 2) this.pinchDist = 0;
      if (this.dragging?.id === e.pointerId) this.dragging = null;
    };
    listen('pointerup', end);
    listen('pointercancel', end);
    listen(
      'wheel',
      (e) => {
        e.preventDefault();
        this.dolly(Math.exp(e.deltaY * 0.001));
      },
      { passive: false },
    );
  }

  private dolly(factor: number): void {
    this.goal.distance = Math.min(this.maxDistance, Math.max(this.minDistance, this.goal.distance * factor));
  }

  /** Re-frames the camera (animated, like user input). */
  setView(v: { target?: [number, number, number]; distance?: number; yaw?: number; pitch?: number }): void {
    if (v.target) vec3.set(...v.target, this.target);
    if (v.distance !== undefined) this.goal.distance = Math.min(this.maxDistance, Math.max(this.minDistance, v.distance));
    if (v.yaw !== undefined) this.goal.yaw = v.yaw;
    if (v.pitch !== undefined) this.goal.pitch = v.pitch;
  }

  get interacting(): boolean {
    return this.dragging !== null || this.pointers.size > 0;
  }

  update(dt: number, aspect: number): void {
    if (!this.interacting && this.autoRotate) this.goal.yaw += this.autoRotate * dt;
    const k = 1 - Math.pow(2, -dt / 0.08);
    this.yaw += (this.goal.yaw - this.yaw) * k;
    this.pitch += (this.goal.pitch - this.pitch) * k;
    this.distance += (this.goal.distance - this.distance) * k;

    const cp = Math.cos(this.pitch);
    vec3.set(
      this.target[0] + this.distance * cp * Math.sin(this.yaw),
      this.target[1] + this.distance * Math.sin(this.pitch),
      this.target[2] + this.distance * cp * Math.cos(this.yaw),
      this.position,
    );
    mat4.lookAt(this.position, this.target, [0, 1, 0], this.view);
    mat4.perspective(this.fovY, aspect, this.near, this.far, this.projection);
    mat4.multiply(this.projection, this.view, this.viewProjection);
  }

  destroy(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
  }
}
