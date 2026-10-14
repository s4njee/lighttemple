import { parseParam, type ParamSchema, type ParamSink, type ParamValues } from './params';

/**
 * The headless `ParamSink`: it records the schema an experiment declares (so a host can render a
 * panel later) and returns live values, but builds no DOM and needs none. An embedder that has no
 * panel passes this where the gallery passes a `ParamsPanel`.
 */
export class HeadlessParams implements ParamSink {
  readonly groups: { title?: string; schema: ParamSchema }[] = [];
  private readonly entries = new Map<string, { get(): unknown }>();
  private readonly changeListeners = new Set<() => void>();
  private readonly buttons: { label: string; onClick: () => void }[] = [];

  constructor(private readonly initial: URLSearchParams = new URLSearchParams()) {}

  add<const S extends ParamSchema>(schema: S, title?: string): ParamValues<S> {
    const values = {} as Record<string, unknown>;
    for (const [key, def] of Object.entries(schema)) {
      if (this.entries.has(key)) throw new Error(`Duplicate param key "${key}"`);
      const raw = this.initial.get(key);
      values[key] = (raw !== null ? parseParam(def, raw) : undefined) ?? def.value;
      this.entries.set(key, { get: () => values[key] });
    }
    this.groups.push({ title, schema });
    return values as ParamValues<S>;
  }

  button(label: string, onClick: () => void): void {
    this.buttons.push({ label, onClick });
  }

  onChange(listener: () => void): () => void {
    this.changeListeners.add(listener);
    return () => this.changeListeners.delete(listener);
  }
}
