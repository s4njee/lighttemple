/**
 * Machine-readable page status on `window.__lighttemple`, polled by the
 * headless smoke test (scripts/smoke.mjs) to know when a frame has rendered
 * or an experiment failed.
 */

export interface PageStatus {
  state: 'index' | 'loading' | 'running' | 'error';
  slug?: string;
  frames: number;
  error?: string;
}

declare global {
  interface Window {
    __lighttemple?: PageStatus;
  }
}

export function reportStatus(status: PageStatus): void {
  window.__lighttemple = status;
}
