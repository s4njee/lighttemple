/**
 * The live HTML the Shadow Catcher floats over: a CSS gradient and a short, real article (selectable
 * text, a working link). Built into the host's backdrop element, behind the transparent canvas.
 * Styles live in src/styles/main.css (`.catcher-*`).
 */

export interface CatcherBackdrop {
  setVariant(variant: string): void;
  dispose(): void;
}

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string): HTMLElementTagNameMap[K] => {
  const node = document.createElement(tag);
  node.className = className;
  if (text) node.textContent = text;
  return node;
};

export function buildBackdrop(host: HTMLElement): CatcherBackdrop {
  const page = el('div', 'catcher-page');
  const article = el('article', 'catcher-article');
  const link = el('a', 'catcher-link', 'Back to the gallery');
  link.href = '#/';
  article.append(
    el('p', 'catcher-eyebrow', 'Field notes · Shadows on the web'),
    el('h2', 'catcher-title', 'An object that casts its shadow on the page'),
    el(
      'p',
      'catcher-lede',
      'Everything behind the object is ordinary HTML: this gradient is CSS, and this paragraph is text you can select and copy. The WebGPU canvas on top is transparent. It draws the object, and on the page only a soft shadow and a little contact darkness, as premultiplied black with alpha equal to how dark each pixel should be.',
    ),
    el(
      'p',
      'catcher-body',
      'The browser blends the canvas over the page, so the shadow falls across headings, words and colour alike, and moves with the object as it turns. Nothing here is a screenshot of the page.',
    ),
    link,
  );
  page.append(article);
  host.append(page);

  let variant = '';
  return {
    setVariant(v) {
      if (v === variant) return;
      variant = v;
      page.dataset.variant = v;
    },
    dispose() {
      page.remove();
    },
  };
}
