import type { PaperSettings } from './types';

export const PAPER_PRESETS: Record<string, Partial<PaperSettings> & { label: string }> = {
  plain:            { label: 'Plain Paper' },
  college:          { label: 'College Ruled', gridSizeMm: 7.1, showMarginLine: true },
  wide:             { label: 'Wide Ruled', gridSizeMm: 8.7, showMarginLine: true },
  narrow:           { label: 'Narrow Ruled', gridSizeMm: 6.4, showMarginLine: true },
  squared5mm:       { label: '5 mm Squared', gridSizeMm: 5 },
  graph:            { label: 'Graph Paper', gridSizeMm: 5 },
  dotgrid:          { label: 'Dot Grid', gridSizeMm: 5 },
  fine:             { label: 'Fine Ruled', gridSizeMm: 5.5, showMarginLine: true },
  'blank-notebook': { label: 'Blank Notebook' },
  custom:           { label: 'Custom Paper' },
};

export const DEFAULT_PAPER: PaperSettings = {
  kind: 'college',
  widthMm: 210,
  heightMm: 297,
  marginMm: 18,
  pageColor: '#fdfcf7',
  lineColor: '#b9cdea',
  lineThickness: 1,
  gridSizeMm: 7.1,
  showMarginLine: true,
};

const MM = 96 / 25.4; // css px per mm

export function paperCss(p: PaperSettings): React.CSSProperties {
  const grid = Math.max(p.gridSizeMm, 2) * MM;
  const t = Math.max(p.lineThickness, 0.5);
  const line = p.lineColor;
  const layers: string[] = [];
  const positions: string[] = [];
  const sizes: string[] = [];

  const ruled = ['college', 'wide', 'narrow', 'fine'].includes(p.kind);
  if (ruled) {
    layers.push(`repeating-linear-gradient(to bottom, transparent, transparent ${grid - t}px, ${line} ${grid - t}px, ${line} ${grid}px)`);
    positions.push('0 0'); sizes.push('auto');
  } else if (p.kind === 'squared5mm' || p.kind === 'graph') {
    layers.push(
      `repeating-linear-gradient(to bottom, transparent, transparent ${grid - t}px, ${line} ${grid - t}px, ${line} ${grid}px)`,
      `repeating-linear-gradient(to right, transparent, transparent ${grid - t}px, ${line} ${grid - t}px, ${line} ${grid}px)`,
    );
    positions.push('0 0', '0 0'); sizes.push('auto', 'auto');
    if (p.kind === 'graph') {
      // heavier line every 5 cells
      const g5 = grid * 5;
      layers.push(
        `repeating-linear-gradient(to bottom, transparent, transparent ${g5 - t * 2}px, ${line} ${g5 - t * 2}px, ${line} ${g5}px)`,
        `repeating-linear-gradient(to right, transparent, transparent ${g5 - t * 2}px, ${line} ${g5 - t * 2}px, ${line} ${g5}px)`,
      );
      positions.push('0 0', '0 0'); sizes.push('auto', 'auto');
    }
  } else if (p.kind === 'dotgrid') {
    layers.push(`radial-gradient(circle at center, ${line} ${t}px, transparent ${t + 0.5}px)`);
    positions.push('0 0'); sizes.push(`${grid}px ${grid}px`);
  }

  return {
    width: `${p.widthMm * MM}px`,
    minHeight: `${p.heightMm * MM}px`,
    backgroundColor: p.pageColor,
    backgroundImage: layers.join(', ') || undefined,
    backgroundPosition: positions.join(', ') || undefined,
    backgroundSize: sizes.join(', ') || undefined,
  };
}
