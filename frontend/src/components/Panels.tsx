import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { DEFAULT_PAPER, PAPER_PRESETS } from '../paper';
import type { BenchmarkReport, PaperSettings, StyleProfile, Typography } from '../types';

// ------------------------------------------------------------------- Styles
export function StylePanel({ styleId, setStyleId }: {
  styleId: string | null; setStyleId: (s: string | null) => void;
}) {
  const [styles, setStyles] = useState<StyleProfile[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = () => api.listStyles().then((s) => {
    setStyles(s);
    if (!styleId && s.length) setStyleId(s[s.length - 1].styleId);
  }).catch((e) => setErr(String(e)));
  useEffect(() => { refresh(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const upload = async (files: FileList) => {
    setBusy(true); setErr('');
    try {
      const p = await api.uploadStyle(Array.from(files), `Style ${styles.length + 1}`);
      setStyleId(p.styleId);
      await refresh();
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  const sel = styles.find((s) => s.styleId === styleId);
  return (
    <section className="panel">
      <h3>1 · Your handwriting</h3>
      <p className="hint">Upload photos/scans of your handwriting. The AI learns the
        complete style zero-shot — no per-user training.</p>
      <button className="primary" disabled={busy} onClick={() => fileRef.current?.click()}>
        {busy ? 'Analyzing…' : 'Upload samples'}
      </button>
      <input ref={fileRef} type="file" accept="image/*" multiple hidden
        onChange={(e) => e.target.files?.length && upload(e.target.files)} />
      {err && <p className="error">{err}</p>}
      {styles.length > 0 && (
        <select value={styleId ?? ''} onChange={(e) => setStyleId(e.target.value)}>
          {styles.map((s) => <option key={s.styleId} value={s.styleId}>{s.name}</option>)}
        </select>
      )}
      {sel && (
        <div className="metrics">
          <span>slant {sel.metrics.slant_deg}°</span>
          <span>stroke {sel.metrics.stroke_width_px}px</span>
          <span>x-height {sel.metrics.x_height_px}px</span>
          <span>drift ±{sel.metrics.baseline_drift_px}px</span>
          <span>{sel.nReferenceLines} reference lines</span>
        </div>
      )}
    </section>
  );
}

// -------------------------------------------------------------------- Paper
export function PaperPanel({ paper, setPaper }: {
  paper: PaperSettings; setPaper: (p: PaperSettings) => void;
}) {
  const set = (patch: Partial<PaperSettings>) => setPaper({ ...paper, ...patch });
  return (
    <section className="panel">
      <h3>Paper</h3>
      <select value={paper.kind} onChange={(e) => {
        const kind = e.target.value as PaperSettings['kind'];
        set({ ...DEFAULT_PAPER, ...PAPER_PRESETS[kind], kind, pageColor: paper.pageColor });
      }}>
        {Object.entries(PAPER_PRESETS).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
      </select>
      <label>Width (mm) <input type="number" value={paper.widthMm} min={80} max={1000}
        onChange={(e) => set({ widthMm: +e.target.value })} /></label>
      <label>Height (mm) <input type="number" value={paper.heightMm} min={80} max={1500}
        onChange={(e) => set({ heightMm: +e.target.value })} /></label>
      <label>Margins (mm) <input type="range" min={0} max={40} value={paper.marginMm}
        onChange={(e) => set({ marginMm: +e.target.value })} /></label>
      <label>Page color <input type="color" value={paper.pageColor}
        onChange={(e) => set({ pageColor: e.target.value })} /></label>
      <label>Line color <input type="color" value={paper.lineColor}
        onChange={(e) => set({ lineColor: e.target.value })} /></label>
      <label>Line thickness <input type="range" min={0.5} max={4} step={0.5} value={paper.lineThickness}
        onChange={(e) => set({ lineThickness: +e.target.value })} /></label>
      <label>Grid size (mm) <input type="range" min={3} max={12} step={0.1} value={paper.gridSizeMm}
        onChange={(e) => set({ gridSizeMm: +e.target.value })} /></label>
    </section>
  );
}

// --------------------------------------------------------------- Typography
export function TypographyPanel({ typo, setTypo }: {
  typo: Typography; setTypo: (t: Typography) => void;
}) {
  const set = (patch: Partial<Typography>) => setTypo({ ...typo, ...patch });
  const Slider = ({ k, label, min, max, step }: { k: keyof Typography; label: string; min: number; max: number; step: number }) => (
    <label>{label} <b>{typo[k]}</b>
      <input type="range" min={min} max={max} step={step} value={typo[k]}
        onChange={(e) => set({ [k]: +e.target.value } as Partial<Typography>)} />
    </label>
  );
  return (
    <section className="panel">
      <h3>Typography</h3>
      <p className="hint">Layout controls — the learned style itself is untouched.</p>
      <Slider k="fontScale" label="Font scale" min={0.5} max={2.5} step={0.05} />
      <Slider k="lineHeight" label="Line height" min={1} max={3} step={0.05} />
      <Slider k="letterSpacing" label="Letter spacing" min={-2} max={12} step={0.5} />
      <Slider k="wordSpacing" label="Word spacing" min={-4} max={30} step={1} />
      <Slider k="paragraphSpacing" label="Paragraph spacing" min={0} max={48} step={2} />
      <Slider k="pagePadding" label="Page padding" min={0} max={64} step={4} />
    </section>
  );
}

// --------------------------------------------------------------- Benchmarks
export function BenchmarkPanel({ styleId }: { styleId: string | null }) {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');

  const run = async () => {
    if (!styleId) return;
    setBusy(true); setErr('');
    try { setReport(await api.runBenchmark(styleId)); }
    catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  };

  const fmt = (v: unknown) => typeof v === 'number' ? v.toFixed(4) : String(v);
  return (
    <section className="panel">
      <h3>Style fidelity benchmarks</h3>
      <p className="hint">Real industry metrics — official HWD (BMVC 2023) and
        writer-retrieval mAP — scoring generated ink against your uploaded samples.</p>
      <button className="primary" disabled={!styleId || busy} onClick={run}>
        {busy ? 'Generating + scoring…' : 'Run benchmark'}
      </button>
      {err && <p className="error">{err}</p>}
      {report && (
        <table className="bench">
          <tbody>
            {Object.entries(report.metrics).map(([name, m]) => (
              <tr key={name}>
                <td>{name.toUpperCase()}</td>
                <td>{'value' in m ? fmt(m.value) : 'mAP' in m ? fmt(m.mAP) : '—'}</td>
                <td className="dir">{String(m.direction ?? '')}</td>
              </tr>
            ))}
            <tr><td>images</td><td>{report.nGeneratedImages} gen / {report.nReferenceImages} real</td><td /></tr>
            <tr><td>engines</td><td colSpan={2}>{report.engines.join(', ')}</td></tr>
          </tbody>
        </table>
      )}
    </section>
  );
}
