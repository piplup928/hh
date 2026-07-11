/**
 * InkOverlay — renders model-generated handwriting over the live document.
 *
 * How real-time editing stays smooth:
 *  - The TipTap/ProseMirror document renders normally (invisible glyphs,
 *    visible caret/selection), so ALL editing features — alignment, justify,
 *    tables, lists, image wrap — keep native browser layout behavior.
 *  - This overlay walks the document, splits every textblock into runs of
 *    identical marks, requests ink for each run over the WebSocket (debounced,
 *    content-addressed cache: retyping/reflowing never re-runs diffusion),
 *    and draws each generated word aligned to its on-screen DOM rect.
 *  - Pending runs are shown as faint placeholder text on the canvas and get
 *    replaced by real ink when the model responds.
 */
import { useEffect, useRef } from 'react';
import type { Editor } from '@tiptap/react';
import { GenSocket } from '../api';
import type { InkPayload, Marks, Typography } from '../types';

interface RunInfo {
  key: string;
  text: string;
  marks: Marks;
  underline: boolean;
  tokens: { text: string; from: number; to: number }[];
}

interface Sprite { img: HTMLImageElement; payloads: InkPayload[] }

const markSig = (m: Marks) => JSON.stringify([!!m.bold, !!m.italic, m.color ?? '', m.sizeFactor ?? 1]);

function collectRuns(editor: Editor, fontScale: number): RunInfo[] {
  const runs: RunInfo[] = [];
  const doc = editor.state.doc;
  doc.descendants((node, pos) => {
    if (!node.isTextblock) return true;
    let cur: RunInfo | null = null;
    node.forEach((child, offset) => {
      if (!child.isText || !child.text) { cur = null; return; }
      const from = pos + 1 + offset;
      const marks: Marks = { sizeFactor: fontScale };
      let underline = false;
      child.marks.forEach((m) => {
        if (m.type.name === 'bold') marks.bold = true;
        if (m.type.name === 'italic') marks.italic = true;
        if (m.type.name === 'underline') underline = true;
        if (m.type.name === 'textStyle' && m.attrs.color) marks.color = m.attrs.color;
        if (m.type.name === 'highlight') { /* drawn client-side */ }
      });
      // headings render larger
      if (node.type.name === 'heading') {
        marks.sizeFactor = (marks.sizeFactor ?? 1) * (node.attrs.level === 1 ? 1.7 : node.attrs.level === 2 ? 1.45 : 1.25);
        marks.bold = true;
      }
      const sig = markSig(marks) + (underline ? 'u' : '');
      if (!cur || cur.key.split('||')[1] !== sig) {
        cur = { key: '', text: '', marks, underline, tokens: [] };
        runs.push(cur);
        cur.key = `||${sig}`;
      }
      // tokenize with positions
      const re = /\S+/g;
      let m2: RegExpExecArray | null;
      while ((m2 = re.exec(child.text)) !== null) {
        cur.tokens.push({ text: m2[0], from: from + m2.index, to: from + m2.index + m2[0].length });
      }
      cur.text = cur.text ? `${cur.text} ${child.text.trim()}` : child.text.trim();
    });
    return true;
  });
  return runs
    .filter((r) => r.tokens.length > 0)
    .map((r) => ({ ...r, key: `${r.text}${r.key}` }));
}

function tokenRect(editor: Editor, from: number, to: number, origin: DOMRect) {
  try {
    const start = editor.view.coordsAtPos(from);
    const end = editor.view.coordsAtPos(to);
    return {
      left: start.left - origin.left,
      top: Math.min(start.top, end.top) - origin.top,
      width: Math.max(end.right ?? end.left, end.left) - start.left,
      height: Math.max(start.bottom, end.bottom) - Math.min(start.top, end.top),
    };
  } catch { return null; }
}

export function useInkOverlay(
  editor: Editor | null,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  containerRef: React.RefObject<HTMLDivElement | null>,
  styleId: string | null,
  typography: Typography,
  inkEnabled: boolean,
) {
  const sprites = useRef(new Map<string, Sprite>());
  const pending = useRef(new Set<string>());
  const keyByReq = useRef(new Map<number, string>());
  const reqCounter = useRef(1);
  const socket = useRef<GenSocket | null>(null);
  const raf = useRef(0);
  const debounce = useRef(0);

  // stable refs for the draw closure
  const stateRef = useRef({ editor, styleId, typography, inkEnabled });
  stateRef.current = { editor, styleId, typography, inkEnabled };

  useEffect(() => {
    socket.current = new GenSocket((reqId, results) => {
      const key = keyByReq.current.get(reqId);
      keyByReq.current.delete(reqId);
      if (!key) return;
      pending.current.delete(key);
      if (!results || !results.length) return;
      const canvases = results.map((p) => {
        const img = new Image();
        img.src = `data:image/png;base64,${p.png}`;
        return img;
      });
      let loaded = 0;
      canvases.forEach((img) => {
        img.onload = () => { if (++loaded === canvases.length) scheduleDraw(); };
      });
      sprites.current.set(key, { img: canvases[0], payloads: results });
      // stash every image on its payload for the draw pass
      results.forEach((p, i) => ((p as InkPayload & { _img?: HTMLImageElement })._img = canvases[i]));
      scheduleDraw();
    });
    return () => socket.current?.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const scheduleDraw = () => {
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(draw);
  };

  const requestMissing = (runs: RunInfo[]) => {
    const { styleId: sid } = stateRef.current;
    if (!sid) return;
    for (const run of runs) {
      const key = `${sid}:${run.key}`;
      if (sprites.current.has(key) || pending.current.has(key)) continue;
      pending.current.add(key);
      const reqId = reqCounter.current++;
      keyByReq.current.set(reqId, key);
      socket.current?.request(reqId, run.text, sid, run.marks);
    }
  };

  const draw = () => {
    const { editor: ed, styleId: sid, typography: typo, inkEnabled: on } = stateRef.current;
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !ed || ed.isDestroyed) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(rect.width * dpr) ||
        canvas.height !== Math.round(rect.height * dpr)) {
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!on || !sid) return;

    const runs = collectRuns(ed, typo.fontScale);
    requestMissing(runs);

    for (const run of runs) {
      const key = `${sid}:${run.key}`;
      const sprite = sprites.current.get(key);
      if (!sprite) { drawPlaceholder(ctx, ed, run, rect); continue; }
      drawRun(ctx, ed, run, sprite.payloads, rect);
    }
  };

  const drawPlaceholder = (ctx: CanvasRenderingContext2D, ed: Editor, run: RunInfo, origin: DOMRect) => {
    ctx.save();
    ctx.fillStyle = 'rgba(60,60,90,0.35)';
    for (const tok of run.tokens) {
      const r = tokenRect(ed, tok.from, tok.to, origin);
      if (!r) continue;
      ctx.font = `italic ${Math.max(r.height * 0.72, 10)}px "Segoe Script", "Comic Sans MS", cursive`;
      ctx.fillText(tok.text, r.left, r.top + r.height * 0.78);
    }
    ctx.restore();
  };

  const drawRun = (
    ctx: CanvasRenderingContext2D, ed: Editor, run: RunInfo,
    payloads: InkPayload[], origin: DOMRect,
  ) => {
    let ti = 0; // token cursor
    for (const p of payloads) {
      const img = (p as InkPayload & { _img?: HTMLImageElement })._img;
      const nTokens = p.text.split(/\s+/).filter(Boolean).length || 1;
      const toks = run.tokens.slice(ti, ti + nTokens);
      ti += nTokens;
      if (!img || !img.complete || toks.length === 0) continue;

      const wordBoxes = p.lines.flatMap((ln) => ln.words.length ? ln.words : [ln]);
      if (wordBoxes.length === toks.length) {
        // word-precise placement
        toks.forEach((tok, i) => {
          const wb = wordBoxes[i];
          const r = tokenRect(ed, tok.from, tok.to, origin);
          if (!r || wb.w < 2 || wb.h < 2) return;
          const scale = r.height / Math.max(wb.h, 1);
          const dw = Math.min(wb.w * scale, r.width * 1.4);
          const dh = wb.h * scale;
          ctx.drawImage(img, wb.x, wb.y, wb.w, wb.h, r.left, r.top + (r.height - dh) / 2, dw, dh);
          if (run.underline) underlineAt(ctx, r, run.marks.color);
        });
      } else {
        // line-strip fallback: group tokens by visual line, stretch ink lines
        const groups: { rects: { left: number; top: number; width: number; height: number }[] }[] = [];
        let lastTop = -1e9;
        toks.forEach((tok) => {
          const r = tokenRect(ed, tok.from, tok.to, origin);
          if (!r) return;
          if (Math.abs(r.top - lastTop) > r.height * 0.5) { groups.push({ rects: [] }); lastTop = r.top; }
          groups[groups.length - 1].rects.push(r);
        });
        const srcLines = p.lines.length ? p.lines : [{ x: 0, y: 0, w: p.width, h: p.height, baseline: p.height, text: p.text, words: [] }];
        groups.forEach((g, gi) => {
          const src = srcLines[Math.min(gi, srcLines.length - 1)];
          if (!g.rects.length || src.w < 2) return;
          const left = Math.min(...g.rects.map((r) => r.left));
          const right = Math.max(...g.rects.map((r) => r.left + r.width));
          const top = Math.min(...g.rects.map((r) => r.top));
          const height = Math.max(...g.rects.map((r) => r.height));
          const scale = height / Math.max(src.h, 1);
          const dw = Math.min(src.w * scale, (right - left) * 1.25);
          ctx.drawImage(img, src.x, src.y, src.w, src.h, left, top, dw, height);
          if (run.underline) underlineAt(ctx, { left, top, width: dw, height }, run.marks.color);
        });
      }
    }
  };

  const underlineAt = (
    ctx: CanvasRenderingContext2D,
    r: { left: number; top: number; width: number; height: number },
    color?: string,
  ) => {
    ctx.save();
    ctx.strokeStyle = color ?? '#1a1a2e';
    ctx.lineWidth = 1.6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    const y = r.top + r.height * 1.02;
    ctx.moveTo(r.left, y);
    const n = Math.max(Math.floor(r.width / 14), 2);
    for (let i = 1; i <= n; i++) {
      ctx.lineTo(r.left + (r.width * i) / n, y + Math.sin(i * 2.1) * 0.9);
    }
    ctx.stroke();
    ctx.restore();
  };

  // redraw triggers
  useEffect(() => {
    if (!editor) return;
    const onUpdate = () => {
      clearTimeout(debounce.current);
      scheduleDraw(); // placeholders update instantly
      debounce.current = window.setTimeout(scheduleDraw, 120);
    };
    editor.on('update', onUpdate);
    editor.on('selectionUpdate', scheduleDraw);
    const ro = new ResizeObserver(scheduleDraw);
    if (containerRef.current) ro.observe(containerRef.current);
    window.addEventListener('resize', scheduleDraw);
    scheduleDraw();
    return () => {
      editor.off('update', onUpdate);
      editor.off('selectionUpdate', scheduleDraw);
      ro.disconnect();
      window.removeEventListener('resize', scheduleDraw);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor, styleId, typography, inkEnabled]);
}
