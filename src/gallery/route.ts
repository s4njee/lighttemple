/**
 * Hash routes (fully static hosting, no server rewrites):
 *   #/                      gallery index
 *   #/x/<slug>              an experiment
 *   #/x/<slug>?k=v&k2=v2    an experiment with non-default params
 */

export type Route = { view: 'index' } | { view: 'experiment'; slug: string; params: URLSearchParams };

export function parseRoute(hash: string): Route {
  const m = hash.match(/^#\/x\/([\w-]+)(?:\?(.*))?$/);
  if (!m) return { view: 'index' };
  return { view: 'experiment', slug: m[1], params: new URLSearchParams(m[2] ?? '') };
}

export function experimentHash(slug: string, params?: URLSearchParams): string {
  const q = params?.toString();
  return `#/x/${slug}${q ? `?${q}` : ''}`;
}

/** Rewrites the current URL without adding history entries or firing hashchange. */
export function replaceHash(hash: string): void {
  if (location.hash !== hash) history.replaceState(history.state, '', hash);
}
