import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import { TextStyleKit } from '@tiptap/extension-text-style';
import Highlight from '@tiptap/extension-highlight';
import TextAlign from '@tiptap/extension-text-align';
import { Table, TableCell, TableHeader, TableRow } from '@tiptap/extension-table';
import { useEffect, useRef } from 'react';
import { HandImage } from './extensions';
import { useInkOverlay } from './InkOverlay';
import type { PaperSettings, Typography } from '../types';
import { paperCss } from '../paper';

interface Props {
  styleId: string | null;
  typo: Typography;
  paper: PaperSettings;
  inkEnabled: boolean;
  onEditor: (e: ReturnType<typeof useEditor>) => void;
}

// Start EMPTY: pre-filled demo text would immediately queue minutes of
// diffusion for content the user is about to delete. Guidance lives in the
// sidebar instead.
const WELCOME = '<p></p>';

export function HandwritingEditor({ styleId, typo, paper, inkEnabled, onEditor }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const editor = useEditor({
    extensions: [
      StarterKit,
      TextStyleKit,
      Highlight.configure({ multicolor: true }),
      TextAlign.configure({ types: ['heading', 'paragraph'] }),
      Table.configure({ resizable: true }),
      TableRow, TableHeader, TableCell,
      HandImage,
    ],
    content: WELCOME,
  });

  useEffect(() => {
    if (editor && !editor.isDestroyed) onEditor(editor);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor]);

  useInkOverlay(editor, canvasRef, containerRef, styleId, typo, inkEnabled);

  const mmPad = paper.marginMm * (96 / 25.4) + typo.pagePadding;
  return (
    <div className="page-scroll">
      <div
        ref={containerRef}
        className={`page ${inkEnabled ? 'ink-on' : ''} ${paper.showMarginLine ? 'margin-line' : ''}`}
        style={{
          ...paperCss(paper),
          padding: `${mmPad}px`,
          ['--hw-font-scale' as string]: typo.fontScale,
          ['--hw-line-height' as string]: typo.lineHeight,
          ['--hw-letter-spacing' as string]: `${typo.letterSpacing}px`,
          ['--hw-word-spacing' as string]: `${typo.wordSpacing}px`,
          ['--hw-para-spacing' as string]: `${typo.paragraphSpacing}px`,
        }}
      >
        <EditorContent editor={editor} />
        <canvas ref={canvasRef} className="ink-canvas" />
      </div>
    </div>
  );
}
