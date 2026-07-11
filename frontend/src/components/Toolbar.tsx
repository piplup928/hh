import type { Editor } from '@tiptap/react';
import { useRef } from 'react';

interface Props {
  editor: Editor | null;
  inkEnabled: boolean;
  setInkEnabled: (v: boolean) => void;
}

const Btn = ({ on, act, label, title }: { on?: boolean; act: () => void; label: string; title: string }) => (
  <button className={`tb ${on ? 'on' : ''}`} onMouseDown={(e) => { e.preventDefault(); act(); }} title={title}>
    {label}
  </button>
);

export function Toolbar({ editor, inkEnabled, setInkEnabled }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  if (!editor || editor.isDestroyed) return null;
  // lazy chain: never invoke commands during render (StrictMode-safe)
  const c = new Proxy({} as ReturnType<Editor['chain']>, {
    get: (_t, prop) => (...args: unknown[]) => {
      const chain = (editor.chain().focus() as unknown as Record<string, (...a: unknown[]) => unknown>);
      return (chain[prop as string](...args));
    },
  });

  const insertImage = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      editor.chain().focus().insertContent({
        type: 'handImage',
        attrs: { src: reader.result as string, width: 320, rotate: 0, wrap: 'inline' },
      }).run();
    };
    reader.readAsDataURL(file);
  };

  const updateImage = (attrs: Record<string, unknown>) => {
    editor.chain().focus().updateAttributes('handImage', attrs).run();
  };
  const imgSelected = editor.isActive('handImage');
  const imgAttrs = imgSelected ? editor.getAttributes('handImage') : null;

  return (
    <div className="toolbar">
      <div className="tb-group">
        <Btn on={editor.isActive('bold')} act={() => c.toggleBold().run()} label="B" title="Bold" />
        <Btn on={editor.isActive('italic')} act={() => c.toggleItalic().run()} label="I" title="Italic" />
        <Btn on={editor.isActive('underline')} act={() => c.toggleUnderline().run()} label="U" title="Underline" />
      </div>
      <div className="tb-group">
        <select className="tb-select" title="Block type"
          value={editor.isActive('heading', { level: 1 }) ? 'h1'
            : editor.isActive('heading', { level: 2 }) ? 'h2'
            : editor.isActive('heading', { level: 3 }) ? 'h3' : 'p'}
          onChange={(e) => {
            const v = e.target.value;
            if (v === 'p') c.setParagraph().run();
            else c.toggleHeading({ level: Number(v[1]) as 1 | 2 | 3 }).run();
          }}>
          <option value="p">Paragraph</option>
          <option value="h1">Heading 1</option>
          <option value="h2">Heading 2</option>
          <option value="h3">Heading 3</option>
        </select>
      </div>
      <div className="tb-group">
        <label className="tb-mini" title="Ink color">🖊
          <input type="color" defaultValue="#1a1a2e"
            onChange={(e) => c.setColor(e.target.value).run()} />
        </label>
        <label className="tb-mini" title="Highlight">🖍
          <input type="color" defaultValue="#fff176"
            onChange={(e) => c.setHighlight({ color: e.target.value }).run()} />
        </label>
        <select className="tb-select" title="Font size" defaultValue=""
          onChange={(e) => e.target.value && c.setFontSize(e.target.value).run()}>
          <option value="" disabled>Size</option>
          {['14px', '16px', '18px', '22px', '26px', '32px', '40px'].map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>
      <div className="tb-group">
        <Btn on={editor.isActive({ textAlign: 'left' })} act={() => c.setTextAlign('left').run()} label="⇤" title="Align left" />
        <Btn on={editor.isActive({ textAlign: 'center' })} act={() => c.setTextAlign('center').run()} label="↔" title="Align center" />
        <Btn on={editor.isActive({ textAlign: 'right' })} act={() => c.setTextAlign('right').run()} label="⇥" title="Align right" />
        <Btn on={editor.isActive({ textAlign: 'justify' })} act={() => c.setTextAlign('justify').run()} label="⇹" title="Justify" />
      </div>
      <div className="tb-group">
        <Btn on={editor.isActive('bulletList')} act={() => c.toggleBulletList().run()} label="•≡" title="Bullet list" />
        <Btn on={editor.isActive('orderedList')} act={() => c.toggleOrderedList().run()} label="1≡" title="Numbered list" />
        <Btn act={() => c.sinkListItem('listItem').run()} label="→≡" title="Indent (multi-level)" />
        <Btn act={() => c.liftListItem('listItem').run()} label="←≡" title="Outdent" />
      </div>
      <div className="tb-group">
        <Btn act={() => c.insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()} label="⊞" title="Insert 3x3 table" />
        <Btn act={() => c.addColumnAfter().run()} label="+col" title="Add column" />
        <Btn act={() => c.addRowAfter().run()} label="+row" title="Add row" />
        <Btn act={() => c.deleteColumn().run()} label="-col" title="Delete column" />
        <Btn act={() => c.deleteRow().run()} label="-row" title="Delete row" />
        <Btn act={() => c.mergeCells().run()} label="⧉" title="Merge cells" />
        <Btn act={() => c.splitCell().run()} label="◫" title="Split cell" />
        <Btn act={() => c.deleteTable().run()} label="⊠" title="Delete table" />
      </div>
      <div className="tb-group">
        <Btn act={() => fileRef.current?.click()} label="🖼" title="Insert image" />
        <input ref={fileRef} type="file" accept="image/*" hidden
          onChange={(e) => e.target.files?.[0] && insertImage(e.target.files[0])} />
        {imgSelected && imgAttrs && (
          <>
            <select className="tb-select" value={imgAttrs.wrap}
              onChange={(e) => updateImage({ wrap: e.target.value })} title="Text wrap">
              <option value="inline">Inline</option>
              <option value="left">Wrap right (img left)</option>
              <option value="right">Wrap left (img right)</option>
            </select>
            <label className="tb-mini" title="Rotate">⟳
              <input type="range" min={-180} max={180} value={imgAttrs.rotate}
                onChange={(e) => updateImage({ rotate: Number(e.target.value) })} />
            </label>
          </>
        )}
      </div>
      <div className="tb-group">
        <Btn act={() => window.print()} label="⎙" title="Print / export PDF" />
        <button className={`tb ink-toggle ${inkEnabled ? 'on' : ''}`}
          onMouseDown={(e) => { e.preventDefault(); setInkEnabled(!inkEnabled); }}
          title="Toggle handwriting rendering">
          ✍ {inkEnabled ? 'Ink on' : 'Ink off'}
        </button>
      </div>
    </div>
  );
}
