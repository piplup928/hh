"""Handwriting style analysis.

Extracts an interpretable style profile from binarized ink:
slant, stroke width (pressure proxy), x-height, inter-line spacing,
inter-word spacing, baseline drift, letter density and ink darkness.

These measurements do NOT drive the generative models directly (the LDM's
style encoder consumes the raw 768x768 style image and learns the style
end-to-end, zero-shot) — they drive everything around the models:
  * the layout engine (line height / word spacing defaults matched to the user)
  * DiffInk's style conditioning vector + stroke rasterization pen model
  * rich-text transforms that must stay consistent with the learned style
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .preprocess import segment_lines, segment_words


@dataclass
class StyleMetrics:
    slant_deg: float = 0.0            # + = right slant
    stroke_width_px: float = 2.0      # mean pen stroke thickness
    stroke_width_std: float = 0.5     # pressure variation proxy
    x_height_px: float = 24.0         # core letter body height
    ascender_ratio: float = 1.7       # (full line height) / x-height
    line_spacing_px: float = 60.0     # baseline-to-baseline
    word_spacing_px: float = 20.0
    intra_word_gap_px: float = 4.0    # avg gap inside words (letter spacing)
    baseline_drift_px: float = 2.0    # std of baseline deviation across a line
    baseline_slope_deg: float = 0.0   # average per-line up/down tendency
    ink_darkness: float = 0.85        # 0..1 mean ink absorbance
    density: float = 0.08             # ink pixels / bounding area
    lines_analyzed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _stroke_width(ink: np.ndarray) -> tuple[float, float]:
    """Stroke width distribution via the distance transform of the ink mask:
    interior maxima of the distance transform ≈ half stroke width."""
    if ink.max() == 0:
        return 2.0, 0.5
    dist = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)
    skel_vals = dist[dist > 0.8]
    if skel_vals.size == 0:
        return 2.0, 0.5
    # use the upper mode: ridge (medial axis) values
    widths = skel_vals * 2.0
    return float(np.median(widths)), float(np.std(widths))


def _slant(ink: np.ndarray) -> float:
    """Dominant slant via image moments of near-vertical stroke segments.

    Uses the classic shear-search: shear the ink by candidate angles and pick
    the angle maximizing the vertical projection profile's energy
    (sharpest columns = strokes made vertical)."""
    ys, xs = np.nonzero(ink)
    if len(xs) < 200:
        return 0.0
    h = ink.shape[0]
    best_angle, best_energy = 0.0, -1.0
    for angle in np.arange(-40, 41, 2.5):
        shear = np.tan(np.deg2rad(angle))
        xsh = xs + (h - ys) * shear
        hist, _ = np.histogram(xsh, bins=max(ink.shape[1] // 3, 16))
        p = hist.astype(np.float64)
        energy = float((p ** 2).sum())
        if energy > best_energy:
            best_energy, best_angle = energy, float(angle)
    return best_angle


def _baseline_stats(ink: np.ndarray, bands: List[tuple]) -> tuple[float, float, float, float]:
    """Per-line baselines via lower envelope regression.

    Returns (x_height, line_spacing, drift_std, slope_deg)."""
    baselines, x_heights, slopes, drifts = [], [], [], []
    for (y0, y1) in bands:
        band = ink[y0:y1]
        cols = np.nonzero(band.sum(axis=0) > 0)[0]
        if cols.size < 10:
            continue
        # lower envelope: for each ink column, lowest ink pixel
        lows, xs_ = [], []
        for x in cols[:: max(1, len(cols) // 200)]:
            col = np.nonzero(band[:, x])[0]
            if col.size:
                lows.append(col.max())
                xs_.append(x)
        if len(lows) < 8:
            continue
        lows_a = np.array(lows, np.float64)
        xs_a = np.array(xs_, np.float64)
        # robust line fit (descenders are outliers below the baseline)
        A = np.vstack([xs_a, np.ones_like(xs_a)]).T
        m, c = np.linalg.lstsq(A, lows_a, rcond=None)[0]
        resid = lows_a - (m * xs_a + c)
        keep = resid < np.percentile(resid, 75)  # drop deep descenders
        if keep.sum() >= 8:
            m, c = np.linalg.lstsq(A[keep], lows_a[keep], rcond=None)[0]
            resid = lows_a[keep] - (m * xs_a[keep] + c)
        baselines.append(y0 + float(np.median(m * xs_a + c)))
        slopes.append(float(np.rad2deg(np.arctan(m))))
        drifts.append(float(np.std(resid)))
        # x-height: histogram of ink row-density inside band; core body = rows
        # above half-max density
        rows = band.sum(axis=1).astype(np.float64)
        if rows.max() > 0:
            core = np.nonzero(rows > rows.max() * 0.4)[0]
            if core.size:
                x_heights.append(float(core.max() - core.min()))
    x_height = float(np.median(x_heights)) if x_heights else 24.0
    if len(baselines) >= 2:
        line_spacing = float(np.median(np.diff(sorted(baselines))))
    else:
        line_spacing = x_height * 2.4
    drift = float(np.median(drifts)) if drifts else 2.0
    slope = float(np.median(slopes)) if slopes else 0.0
    return x_height, line_spacing, drift, slope


def _spacing(ink: np.ndarray, bands: List[tuple]) -> tuple[float, float]:
    word_gaps, intra_gaps = [], []
    for (y0, y1) in bands:
        band = ink[y0:y1]
        h = band.shape[0]
        words = segment_words(band)
        for i in range(1, len(words)):
            word_gaps.append(words[i][0] - words[i - 1][1])
        # intra-word gaps: ink-column runs inside each word span
        for (s, e) in words:
            proj = band[:, s:e].sum(axis=0) > 0
            gap = 0
            for v in proj:
                if not v:
                    gap += 1
                elif gap:
                    if gap < h * 0.55:
                        intra_gaps.append(gap)
                    gap = 0
    ws = float(np.median(word_gaps)) if word_gaps else 20.0
    ig = float(np.median(intra_gaps)) if intra_gaps else 3.0
    return ws, ig


def analyze(gray: np.ndarray, ink: np.ndarray) -> StyleMetrics:
    bands = segment_lines(ink)
    sw, sw_std = _stroke_width(ink)
    slant = _slant(ink)
    x_h, line_sp, drift, slope = _baseline_stats(ink, bands)
    word_sp, intra = _spacing(ink, bands)

    ys, xs = np.nonzero(ink)
    if len(xs) > 10:
        area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        density = float(len(xs)) / max(float(area), 1.0)
        darkness = float(1.0 - gray[ink > 0].astype(np.float64).mean() / 255.0)
    else:
        density, darkness = 0.08, 0.85

    band_h = np.median([(b - a) for a, b in bands]) if bands else x_h * 1.7
    return StyleMetrics(
        slant_deg=round(slant, 2),
        stroke_width_px=round(sw, 2),
        stroke_width_std=round(sw_std, 2),
        x_height_px=round(x_h, 1),
        ascender_ratio=round(float(band_h) / max(x_h, 1.0), 2),
        line_spacing_px=round(line_sp, 1),
        word_spacing_px=round(word_sp, 1),
        intra_word_gap_px=round(intra, 1),
        baseline_drift_px=round(drift, 2),
        baseline_slope_deg=round(slope, 2),
        ink_darkness=round(darkness, 3),
        density=round(density, 4),
        lines_analyzed=len(bands),
    )


def merge_metrics(all_metrics: List[StyleMetrics]) -> StyleMetrics:
    """Aggregate metrics across multiple uploaded samples (median-combine)."""
    if not all_metrics:
        return StyleMetrics()
    if len(all_metrics) == 1:
        return all_metrics[0]
    fields = StyleMetrics().to_dict().keys()
    agg = {}
    for f in fields:
        vals = [getattr(m, f) for m in all_metrics]
        agg[f] = type(vals[0])(np.median(vals)) if f != "lines_analyzed" else int(sum(vals))
    return StyleMetrics(**agg)
