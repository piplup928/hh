import { useState } from 'react';
import type { Editor } from '@tiptap/react';
import { HandwritingEditor } from './editor/HandwritingEditor';
import { Toolbar } from './components/Toolbar';
import { BenchmarkPanel, PaperPanel, StylePanel, TypographyPanel } from './components/Panels';
import { DEFAULT_PAPER } from './paper';
import type { PaperSettings, Typography } from './types';
import './styles.css';

const DEFAULT_TYPO: Typography = {
  fontScale: 1, lineHeight: 1.8, letterSpacing: 0, wordSpacing: 4,
  paragraphSpacing: 12, pagePadding: 8,
};

export default function App() {
  const [styleId, setStyleId] = useState<string | null>(null);
  const [paper, setPaper] = useState<PaperSettings>(DEFAULT_PAPER);
  const [typo, setTypo] = useState<Typography>(DEFAULT_TYPO);
  const [inkEnabled, setInkEnabled] = useState(true);
  const [editor, setEditor] = useState<Editor | null>(null);

  return (
    <div className="app">
      <aside className="sidebar">
        <h2>✍ Handwriting AI</h2>
        <StylePanel styleId={styleId} setStyleId={setStyleId} />
        <PaperPanel paper={paper} setPaper={setPaper} />
        <TypographyPanel typo={typo} setTypo={setTypo} />
        <BenchmarkPanel styleId={styleId} />
      </aside>
      <main className="workspace">
        <Toolbar editor={editor} inkEnabled={inkEnabled} setInkEnabled={setInkEnabled} />
        <HandwritingEditor
          styleId={styleId} typo={typo} paper={paper} inkEnabled={inkEnabled}
          onEditor={(e) => setEditor(e as Editor)}
        />
      </main>
    </div>
  );
}
