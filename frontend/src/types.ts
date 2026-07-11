export interface WordBox {
  text: string; x: number; y: number; w: number; h: number; baseline: number;
}
export interface LineBox extends WordBox { words: WordBox[] }

export interface InkPayload {
  engine: 'paragraph_ldm' | 'diffink' | 'preview';
  text: string;
  png: string;            // base64
  svg?: string | null;
  width: number;
  height: number;
  xHeight: number;
  seed?: number | null;
  lines: LineBox[];
  meta: Record<string, unknown>;
}

export interface Marks {
  bold?: boolean;
  italic?: boolean;
  color?: string;
  sizeFactor?: number;
}

export interface StyleProfile {
  styleId: string;
  name: string;
  createdAt: number;
  nSamples: number;
  nReferenceLines: number;
  metrics: {
    slant_deg: number; stroke_width_px: number; stroke_width_std: number;
    x_height_px: number; ascender_ratio: number; line_spacing_px: number;
    word_spacing_px: number; intra_word_gap_px: number;
    baseline_drift_px: number; baseline_slope_deg: number;
    ink_darkness: number; density: number; lines_analyzed: number;
  };
}

export type PaperKind =
  | 'plain' | 'college' | 'wide' | 'narrow' | 'squared5mm' | 'graph'
  | 'dotgrid' | 'fine' | 'blank-notebook' | 'custom';

export interface PaperSettings {
  kind: PaperKind;
  widthMm: number;
  heightMm: number;
  marginMm: number;
  pageColor: string;
  lineColor: string;
  lineThickness: number;  // px
  gridSizeMm: number;
  showMarginLine: boolean;
}

export interface Typography {
  fontScale: number;       // multiplies handwriting size
  lineHeight: number;      // multiplier
  letterSpacing: number;   // px
  wordSpacing: number;     // px
  paragraphSpacing: number;// px
  pagePadding: number;     // px extra inside margins
}

export interface BenchmarkReport {
  runId: string;
  styleId: string;
  engines: string[];
  nGeneratedImages: number;
  nReferenceImages: number;
  generationSeconds: number;
  totalSeconds: number;
  metrics: Record<string, Record<string, unknown>>;
}
