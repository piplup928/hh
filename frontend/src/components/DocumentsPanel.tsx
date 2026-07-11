/**
 * Document persistence: save / open / delete + 8s autosave of the TipTap JSON
 * together with the paper, typography and style settings it was written with.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Editor } from '@tiptap/react';
import { api } from '../api';
import type { DocumentSummary, PaperSettings, Typography } from '../types';

interface Props {
  editor: Editor | null;
  styleId: string | null;
  paper: PaperSettings;
  typo: Typography;
  applyLoaded: (doc: {
    paper?: Partial<PaperSettings>; typography?: Partial<Typography>;
    styleId?: string | null;
  }) => void;
}

export function DocumentsPanel({ editor, styleId, paper, typo, applyLoaded }: Props) {
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [docId, setDocId] = useState<string | null>(null);
  const [title, setTitle] = useState('Untitled');
  const [status, setStatus] = useState('');
  const dirty = useRef(false);
  const saving = useRef(false);

  const refresh = useCallback(() => {
    api.listDocuments().then(setDocs).catch(() => {});
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (!editor) return;
    const mark = () => { dirty.current = true; };
    editor.on('update', mark);
    return () => { editor.off('update', mark); };
  }, [editor]);

  const save = useCallback(async (auto = false) => {
    if (!editor || editor.isDestroyed || saving.current) return;
    saving.current = true;
    try {
      const r = await api.saveDocument({
        docId, title, content: editor.getJSON() as Record<string, unknown>,
        paper, typography: typo, styleId,
      });
      setDocId(r.docId);
      dirty.current = false;
      setStatus(`${auto ? 'auto' : ''}saved ${new Date(r.updatedAt * 1000).toLocaleTimeString()}`);
      refresh();
    } catch (e) {
      setStatus(`save failed: ${String(e).slice(0, 60)}`);
    } finally {
      saving.current = false;
    }
  }, [editor, docId, title, paper, typo, styleId, refresh]);

  // autosave loop
  const saveRef = useRef(save);
  saveRef.current = save;
  useEffect(() => {
    const t = setInterval(() => { if (dirty.current) saveRef.current(true); }, 8000);
    return () => clearInterval(t);
  }, []);

  const open = async (id: string) => {
    if (!editor || !id) return;
    try {
      const d = await api.getDocument(id);
      editor.commands.setContent(d.content as never);
      setDocId(d.docId ?? id);
      setTitle(d.title || 'Untitled');
      applyLoaded({ paper: d.paper, typography: d.typography, styleId: d.styleId });
      dirty.current = false;
      setStatus('opened');
    } catch (e) { setStatus(`open failed: ${String(e).slice(0, 60)}`); }
  };

  const newDoc = () => {
    if (!editor) return;
    editor.commands.setContent('<p></p>');
    setDocId(null);
    setTitle('Untitled');
    dirty.current = false;
    setStatus('new document');
  };

  const remove = async () => {
    if (!docId) return;
    await api.deleteDocument(docId).catch(() => {});
    newDoc();
    refresh();
  };

  return (
    <section className="panel">
      <h3>Documents</h3>
      <input
        className="doc-title" value={title} placeholder="Title"
        onChange={(e) => { setTitle(e.target.value); dirty.current = true; }}
      />
      <div className="doc-actions">
        <button className="primary" onClick={() => save(false)}>Save</button>
        <button onClick={newDoc}>New</button>
        {docId && <button onClick={remove}>Delete</button>}
      </div>
      <select value="" onChange={(e) => open(e.target.value)}>
        <option value="" disabled>Open document…</option>
        {docs.map((d) => (
          <option key={d.docId} value={d.docId}>
            {d.title || d.docId}
            {d.updatedAt ? ` · ${new Date(d.updatedAt * 1000).toLocaleDateString()}` : ''}
          </option>
        ))}
      </select>
      {status && <p className="hint">{status}</p>}
    </section>
  );
}
