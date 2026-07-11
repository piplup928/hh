/**
 * Custom TipTap extensions: resizable/rotatable/wrappable images.
 * Spacing controls (letter/word/paragraph/line) are applied as CSS variables
 * on the page container — see App.tsx — so they modify layout without
 * touching the learned handwriting style.
 */
import Image from '@tiptap/extension-image';

export const HandImage = Image.extend({
  name: 'handImage',
  addAttributes() {
    return {
      ...this.parent?.(),
      width: { default: 320 },
      rotate: { default: 0 },
      wrap: { default: 'inline' }, // inline | left | right
    };
  },
  renderHTML({ HTMLAttributes }) {
    const { width, rotate, wrap, ...rest } = HTMLAttributes;
    const float = wrap === 'left' ? 'float:left;margin:4px 16px 8px 0;'
      : wrap === 'right' ? 'float:right;margin:4px 0 8px 16px;' : '';
    return ['img', {
      ...rest,
      style: `width:${width}px;transform:rotate(${rotate}deg);${float}`,
      'data-wrap': wrap,
    }];
  },
  addNodeView() {
    return ({ node, editor, getPos }) => {
      const wrapper = document.createElement('span');
      wrapper.className = 'hand-image-wrapper';
      const img = document.createElement('img');
      img.src = node.attrs.src;
      const apply = (n = node) => {
        img.style.width = `${n.attrs.width}px`;
        img.style.transform = `rotate(${n.attrs.rotate}deg)`;
        wrapper.dataset.wrap = n.attrs.wrap;
        wrapper.style.cssFloat =
          n.attrs.wrap === 'left' ? 'left' : n.attrs.wrap === 'right' ? 'right' : '';
        wrapper.style.margin = n.attrs.wrap === 'left' ? '4px 16px 8px 0'
          : n.attrs.wrap === 'right' ? '4px 0 8px 16px' : '0';
      };
      apply();
      wrapper.appendChild(img);

      // resize handle
      const handle = document.createElement('span');
      handle.className = 'hand-image-resize';
      wrapper.appendChild(handle);
      handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const startX = e.clientX;
        const startW = img.getBoundingClientRect().width;
        const move = (ev: MouseEvent) => {
          const w = Math.max(startW + (ev.clientX - startX), 48);
          img.style.width = `${w}px`;
        };
        const up = (ev: MouseEvent) => {
          document.removeEventListener('mousemove', move);
          document.removeEventListener('mouseup', up);
          const w = Math.max(startW + (ev.clientX - startX), 48);
          const pos = typeof getPos === 'function' ? getPos() : null;
          if (pos != null) {
            editor.view.dispatch(editor.view.state.tr.setNodeMarkup(pos, undefined, {
              ...node.attrs, width: Math.round(w),
            }));
          }
        };
        document.addEventListener('mousemove', move);
        document.addEventListener('mouseup', up);
      });

      return {
        dom: wrapper,
        update(updated) {
          if (updated.type.name !== 'handImage') return false;
          img.src = updated.attrs.src;
          apply(updated);
          return true;
        },
      };
    };
  },
});
