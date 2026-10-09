/**
 * Pointer state relative to a canvas, with critically-damped smoothing.
 *
 * Pointer-driven lights are the bread and butter of web lighting effects; a
 * little smoothing makes them feel physical instead of glued to the cursor.
 */

export interface PointerState {
  /** Raw position in [0,1]^2, origin top-left. */
  uv: [number, number];
  /** Smoothed position in [0,1]^2, origin top-left. */
  smoothUv: [number, number];
  /** Smoothed velocity in uv/sec. */
  velocity: [number, number];
  /** Whether the pointer is currently over the canvas. */
  inside: boolean;
  /** Whether a button/touch is held. */
  down: boolean;
  /** Seconds since the pointer last moved (for idle animations). */
  idle: number;
}

export class PointerTracker {
  readonly state: PointerState = {
    uv: [0.5, 0.4],
    smoothUv: [0.5, 0.4],
    velocity: [0, 0],
    inside: false,
    down: false,
    idle: Infinity,
  };
  /** Smoothing half-life in seconds. 0 disables smoothing. */
  halfLife = 0.06;

  private disposers: (() => void)[] = [];

  constructor(private readonly target: HTMLElement) {
    const on = <K extends keyof HTMLElementEventMap>(type: K, fn: (e: HTMLElementEventMap[K]) => void) => {
      target.addEventListener(type, fn, { passive: true });
      this.disposers.push(() => target.removeEventListener(type, fn));
    };
    on('pointermove', (e) => this.move(e));
    on('pointerdown', (e) => {
      this.state.down = true;
      this.move(e);
    });
    on('pointerup', () => (this.state.down = false));
    on('pointercancel', () => (this.state.down = false));
    on('pointerenter', () => (this.state.inside = true));
    on('pointerleave', () => {
      this.state.inside = false;
      this.state.down = false;
    });
  }

  private move(e: PointerEvent): void {
    const r = this.target.getBoundingClientRect();
    this.state.uv = [(e.clientX - r.left) / Math.max(1, r.width), (e.clientY - r.top) / Math.max(1, r.height)];
    this.state.inside = true;
    this.state.idle = 0;
  }

  update(dt: number): void {
    const s = this.state;
    s.idle += dt;
    const k = this.halfLife > 0 ? 1 - Math.pow(2, -dt / this.halfLife) : 1;
    for (const i of [0, 1] as const) {
      const prev = s.smoothUv[i];
      s.smoothUv[i] += (s.uv[i] - prev) * k;
      const v = dt > 0 ? (s.smoothUv[i] - prev) / dt : 0;
      s.velocity[i] += (v - s.velocity[i]) * k;
    }
  }

  destroy(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
  }
}
