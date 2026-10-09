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

interface Entry {
  def: ParamDef;
  /** Updates the live value and the control without notifying listeners. */
  set(v: unknown): void;
  get(): unknown;
}

/** Parses a URL string into a valid value for `def`, or undefined if it doesn't fit. */
function parseParam(def: ParamDef, raw: string): unknown {
  if ('options' in def) return Object.values(def.options).find((v) => String(v) === raw);
  if ('type' in def && def.type === 'color') return /^[0-9a-f]{6}$/i.test(raw) ? `#${raw.toLowerCase()}` : undefined;
  if (typeof def.value === 'boolean') return raw === '1' || raw === 'true' ? true : raw === '0' || raw === 'false' ? false : undefined;
  const d = def as NumberParam;
  const n = Number(raw);
  return raw !== '' && Number.isFinite(n) ? Math.min(d.max, Math.max(d.min, n)) : undefined;
}

function formatParam(def: ParamDef, v: unknown): string {
  if ('type' in def && def.type === 'color') return String(v).slice(1).toLowerCase();
  if (typeof v === 'boolean') return v ? '1' : '0';
  return String(v);
}

export class ParamsPanel {
  readonly element: HTMLElement;
  private entries = new Map<string, Entry>();
  private changeListeners = new Set<() => void>();

  /**
   * @param initial Optional overrides (e.g. from the URL). Keys that don't match
   *   a declared param, or values that don't parse, are ignored.
   */
  constructor(
    container: HTMLElement,
    private readonly initial: URLSearchParams = new URLSearchParams(),
  ) {
    this.element = document.createElement('div');
    this.element.className = 'params';
    container.append(this.element);
  }

  /**
   * Adds a group of controls and returns the live values object.
   * Keys must be unique across the whole panel (they double as URL keys).
   */
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
      if (this.entries.has(key)) throw new Error(`Duplicate param key "${key}"`);
      const raw = this.initial.get(key);
      const start = (raw !== null ? parseParam(def, raw) : undefined) ?? def.value;
      values[key] = start;

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
      let setUi: (v: unknown) => void;

      if ('options' in def) {
        const select = document.createElement('select');
        for (const [label, v] of Object.entries(def.options)) {
          const opt = document.createElement('option');
          opt.textContent = label;
          opt.value = String(v);
          select.append(opt);
        }
        select.addEventListener('change', () => commit(parseParam(def, select.value) ?? def.value));
        setUi = (v) => (select.value = String(v));
        row.append(select);
      } else if ('type' in def && def.type === 'color') {
        const input = document.createElement('input');
        input.type = 'color';
        input.addEventListener('input', () => commit(input.value));
        setUi = (v) => (input.value = String(v));
        row.append(input);
      } else if (typeof def.value === 'boolean') {
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.addEventListener('change', () => commit(input.checked));
        setUi = (v) => (input.checked = Boolean(v));
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
        const out = document.createElement('output');
        const fmt = (v: number) => v.toFixed(Math.min(4, decimals(step)));
        input.addEventListener('input', () => {
          const v = Number(input.value);
          out.textContent = fmt(v);
          commit(v);
        });
        setUi = (v) => {
          input.value = String(v);
          out.textContent = fmt(Number(v));
        };
        row.append(input, out);
      }

      setUi(start);
      this.entries.set(key, {
        def,
        get: () => values[key],
        set: (v) => {
          values[key] = v;
          setUi(v);
        },
      });
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

  /** Only the params that differ from their defaults, in declaration order. */
  serialize(): URLSearchParams {
    const out = new URLSearchParams();
    for (const [key, e] of this.entries) {
      const v = formatParam(e.def, e.get());
      if (v !== formatParam(e.def, e.def.value)) out.set(key, v);
    }
    return out;
  }

  reset(): void {
    for (const e of this.entries.values()) e.set(e.def.value);
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
