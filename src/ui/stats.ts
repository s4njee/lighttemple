/**
 * Minimal frame-time readout. CPU-side rAF timing for now; GPU timestamp
 * queries are a planned upgrade (see plan.md, E9).
 */

export class FrameStats {
  readonly element: HTMLElement;
  private samples: number[] = [];
  private lastPaint = 0;

  constructor(container: HTMLElement) {
    this.element = document.createElement('div');
    this.element.className = 'stats';
    this.element.setAttribute('aria-hidden', 'true');
    container.append(this.element);
  }

  push(dtSeconds: number, now: number, resolution: string): void {
    this.samples.push(dtSeconds * 1000);
    if (this.samples.length > 120) this.samples.shift();
    if (now - this.lastPaint < 250) return;
    this.lastPaint = now;
    const avg = this.samples.reduce((a, b) => a + b, 0) / this.samples.length;
    const worst = Math.max(...this.samples);
    this.element.textContent = `${(1000 / avg).toFixed(0)} fps · ${avg.toFixed(1)} ms (max ${worst.toFixed(1)}) · ${resolution}`;
  }

  destroy(): void {
    this.element.remove();
  }
}
