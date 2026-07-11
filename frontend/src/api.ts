import type { BenchmarkReport, InkPayload, Marks, StyleProfile } from './types';

const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

async function jfetch<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, init);
  if (!r.ok) {
    const body = await r.text().catch(() => '');
    throw new Error(`${r.status} ${path}: ${body.slice(0, 300)}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => jfetch<{ ok: boolean; mode: string }>('/api/health'),
  engineStatus: () => jfetch<Record<string, unknown>>('/api/generate/status'),

  listStyles: () => jfetch<StyleProfile[]>('/api/styles'),
  uploadStyle: async (files: File[], name: string) => {
    const fd = new FormData();
    files.forEach((f) => fd.append('files', f));
    fd.append('name', name);
    return jfetch<StyleProfile>('/api/styles', { method: 'POST', body: fd });
  },
  deleteStyle: (id: string) =>
    jfetch(`/api/styles/${id}`, { method: 'DELETE' }),

  generateText: (text: string, styleId: string, marks?: Marks, quality = 'live') =>
    jfetch<{ results: InkPayload[] }>('/api/generate/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, styleId, marks, quality }),
    }),

  runBenchmark: (styleId: string, maxSamples = 8) =>
    jfetch<BenchmarkReport>('/api/benchmarks/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ styleId, maxSamples }),
    }),
  listReports: () => jfetch<BenchmarkReport[]>('/api/benchmarks/reports'),
};

// ------------------------------------------------------------------ WebSocket
type InkHandler = (reqId: number, results: InkPayload[] | null, err?: string) => void;

export class GenSocket {
  private ws: WebSocket | null = null;
  private handler: InkHandler;
  private queue: string[] = [];
  private reconnectDelay = 500;

  constructor(handler: InkHandler) {
    this.handler = handler;
    this.connect();
  }

  private connect() {
    const url = BASE.replace(/^http/, 'ws') + '/api/generate/ws';
    try {
      this.ws = new WebSocket(url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws.onopen = () => {
      this.reconnectDelay = 500;
      this.queue.splice(0).forEach((m) => this.ws?.send(m));
    };
    this.ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.op === 'ink') this.handler(msg.reqId, msg.results);
        else if (msg.op === 'error') this.handler(msg.reqId, null, msg.detail);
      } catch { /* ignore malformed frames */ }
    };
    this.ws.onclose = () => this.scheduleReconnect();
    this.ws.onerror = () => this.ws?.close();
  }

  private scheduleReconnect() {
    setTimeout(() => this.connect(), this.reconnectDelay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, 8000);
  }

  request(reqId: number, text: string, styleId: string, marks?: Marks) {
    const msg = JSON.stringify({ op: 'gen', reqId, text, styleId, marks, quality: 'live' });
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(msg);
    else this.queue.push(msg);
  }

  close() { this.ws?.close(); }
}
