/**
 * A tiny declarative parameter panel.
 *
 * Experiments declare a schema; they get back a plain, live-updating values
 * object to read every frame. No framework, no dependency, ~zero overhead.
 *
 *   const p = panel.add({
 *     intensity: { value: 2, min: 0, max: 10 },
 *     tint: { value: '#ffd9a0', type: 'color' },
 *     animate: { value: true },
 *     mode: { value: 'pcf', options: { Hard: 'hard', PCF: 'pcf' } },
 *   });
 *   p.intensity // number
 */

export interface NumberParam {
  value: number;
  min: number;
  max: number;
  step?: number;
  label?: string;
  hint?: string;
}
export interface BooleanParam {
  value: boolean;
  label?: string;
  hint?: string;
}
export interface ColorParam {
  value: string;
  type: 'color';
  label?: string;
  hint?: string;
}
export interface SelectParam<T extends string | number = string | number> {
  value: T;
  options: Record<string, T>;
  label?: string;
  hint?: string;
}

export type ParamDef = NumberParam | BooleanParam | ColorParam | SelectParam;
export type ParamSchema = Record<string, ParamDef>;

export type ParamValues<S extends ParamSchema> = {
  [K in keyof S]: S[K] extends NumberParam
    ? number
    : S[K] extends BooleanParam
      ? boolean
      : S[K] extends ColorParam
        ? string
        : S[K] extends SelectParam<infer T>
          ? T
          : never;
};

function humanize(key: string): string {
  return key.replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/^./, (c) => c.toUpperCase());
}

function decimals(step: number): number {
  const s = String(step);
  return s.includes('.') ? s.length - s.indexOf('.') - 1 : 0;
}

export class ParamsPanel {
  readonly element: HTMLElement;
  private resetters: (() => void)[] = [];
  private changeListeners = new Set<() => void>();

  constructor(container: HTMLElement) {
    this.element = document.createElement('div');
    this.element.className = 'params';
    container.append(this.element);
  }

  /** Adds a group of controls and returns the live values object. */
  add<const S extends ParamSchema>(schema: S, title?: string): ParamValues<S> {
    const values = {} as Record<string, unknown>;
    const group = document.createElement('fieldset');
    group.className = 'params-group';
    if (title) {
      const legend = document.createElement('legend');
      legend.textContent = title;
      group.append(legend);
    }

    for (const [key, def] of Object.entries(schema)) {
      values[key] = def.value;
      const row = document.createElement('label');
      row.className = 'param';
      if (def.hint) row.title = def.hint;
      const name = document.createElement('span');
      name.className = 'param-name';
      name.textContent = def.label ?? humanize(key);
      row.append(name);

      const commit = (v: unknown) => {
        values[key] = v;
        for (const l of this.changeListeners) l();
      };

      if ('options' in def) {
        const select = document.createElement('select');
        for (const [label, v] of Object.entries(def.options)) {
          const opt = document.createElement('option');
          opt.textContent = label;
          opt.value = String(v);
          select.append(opt);
        }
        const fromString = (s: string) => Object.values(def.options).find((v) => String(v) === s) ?? def.value;
        select.value = String(def.value);
        select.addEventListener('change', () => commit(fromString(select.value)));
        this.resetters.push(() => {
          select.value = String(def.value);
          values[key] = def.value;
        });
        row.append(select);
      } else if ('type' in def && def.type === 'color') {
        const input = document.createElement('input');
        input.type = 'color';
        input.value = def.value;
        input.addEventListener('input', () => commit(input.value));
        this.resetters.push(() => {
          input.value = def.value;
          values[key] = def.value;
        });
        row.append(input);
      } else if (typeof def.value === 'boolean') {
        const d = def as BooleanParam;
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = d.value;
        input.addEventListener('change', () => commit(input.checked));
        this.resetters.push(() => {
          input.checked = d.value;
          values[key] = d.value;
        });
        row.classList.add('param-bool');
        row.append(input);
      } else {
        const d = def as NumberParam;
        const step = d.step ?? (d.max - d.min) / 100;
        const input = document.createElement('input');
        input.type = 'range';
        input.min = String(d.min);
        input.max = String(d.max);
        input.step = String(step);
        input.value = String(d.value);
        const out = document.createElement('output');
        const fmt = (v: number) => v.toFixed(Math.min(4, decimals(step)));
        out.textContent = fmt(d.value);
        input.addEventListener('input', () => {
          const v = Number(input.value);
          out.textContent = fmt(v);
          commit(v);
        });
        this.resetters.push(() => {
          input.value = String(d.value);
          out.textContent = fmt(d.value);
          values[key] = d.value;
        });
        row.append(input, out);
      }
      group.append(row);
    }

    this.element.append(group);
    return values as ParamValues<S>;
  }

  /** Adds a push button (e.g. "Randomise", "Reset camera"). */
  button(label: string, onClick: () => void): void {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'param-button';
    btn.textContent = label;
    btn.addEventListener('click', onClick);
    this.element.append(btn);
  }

  onChange(listener: () => void): () => void {
    this.changeListeners.add(listener);
    return () => this.changeListeners.delete(listener);
  }

  reset(): void {
    for (const r of this.resetters) r();
    for (const l of this.changeListeners) l();
  }

  destroy(): void {
    this.changeListeners.clear();
    this.element.remove();
  }
}

export function srgbToLinear(s: number): number {
  return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
}

export function linearToSrgb(c: number): number {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

/** Parses '#rrggbb' into linear-light RGB (what shaders want for light colours). */
export function hexToLinear(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [srgbToLinear(((n >> 16) & 255) / 255), srgbToLinear(((n >> 8) & 255) / 255), srgbToLinear((n & 255) / 255)];
}
