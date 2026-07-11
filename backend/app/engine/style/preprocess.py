"""Handwriting sample preprocessing.

Turns arbitrary user uploads (photos/scans of handwritten pages) into the
768x768 grayscale style canvases the Paragraph-LDM style encoder expects,
plus clean binarized ink maps used by the style analyzer and benchmarks.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from ...config import LDM_CANVAS


def to_grayscale(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("L"), dtype=np.uint8)
    return arr


def estimate_background(gray: np.ndarray) -> int:
    """Robust paper-white estimate (median of the brightest half)."""
    flat = gray.reshape(-1)
    bright = flat[flat >= np.median(flat)]
    return int(np.median(bright)) if bright.size else 255


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Remove shadows / uneven lighting from phone photos via large-kernel division."""
    k = max(gray.shape) // 8 | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = np.maximum(bg, 1)
    norm = np.clip(gray.astype(np.float32) / bg.astype(np.float32) * 255.0, 0, 255)
    return norm.astype(np.uint8)


def binarize_ink(gray: np.ndarray) -> np.ndarray:
    """Ink mask (255 = ink). Sauvola-like adaptive threshold works for pen strokes."""
    flat = flatten_illumination(gray)
    th = cv2.adaptiveThreshold(
        flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    # kill salt noise but keep thin strokes
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return th


def deskew(gray: np.ndarray, ink: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Deskew page rotation using the ink minAreaRect / projection entropy."""
    ys, xs = np.nonzero(ink)
    if len(xs) < 100:
        return gray, ink, 0.0
    coords = np.column_stack([xs, ys]).astype(np.float32)
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    angle = float(np.clip(angle, -15, 15))
    if abs(angle) < 0.3:
        return gray, ink, 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    bgv = estimate_background(gray)
    gray_r = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=int(bgv))
    ink_r = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return gray_r, ink_r, angle


def crop_to_ink(gray: np.ndarray, ink: np.ndarray, pad: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(ink)
    if len(xs) < 10:
        return gray, ink
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, gray.shape[1])
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, gray.shape[0])
    return gray[y0:y1, x0:x1], ink[y0:y1, x0:x1]


def make_style_canvas(gray: np.ndarray, ink: np.ndarray, size: int = LDM_CANVAS) -> Image.Image:
    """Fit the cleaned sample onto a white size x size canvas (aspect preserved,
    top-left anchored like the IAM paragraphs the LDM was trained on)."""
    g, m = crop_to_ink(gray, ink)
    h, w = g.shape
    scale = min(size / w, size / h, 1.5)
    nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    # re-normalize contrast: paper -> 255, ink kept
    bgv = estimate_background(g)
    g = np.clip(g.astype(np.float32) * (255.0 / max(bgv, 1)), 0, 255).astype(np.uint8)
    canvas = np.full((size, size), 255, np.uint8)
    ox, oy = (0, 0)
    canvas[oy:oy + nh, ox:ox + nw] = g
    return Image.fromarray(canvas)


def segment_lines(ink: np.ndarray, min_height: int = 8) -> List[Tuple[int, int]]:
    """Text-line segmentation via smoothed horizontal projection profile.

    Returns list of (y0, y1) line bands.
    """
    proj = ink.astype(np.float32).sum(axis=1)
    if proj.max() <= 0:
        return []
    k = max(len(proj) // 200, 3)
    kernel = np.ones(k, np.float32) / k
    smooth = np.convolve(proj, kernel, mode="same")
    thr = smooth.max() * 0.05
    on = smooth > thr
    bands: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_height:
                bands.append((start, i))
            start = None
    if start is not None and len(on) - start >= min_height:
        bands.append((start, len(on)))
    return bands


def segment_words(ink_line: np.ndarray, gap_factor: float = 0.55) -> List[Tuple[int, int]]:
    """Word segmentation of a single line band via vertical whitespace valleys.

    gap threshold is adaptive: gaps wider than `gap_factor * line_height`
    are treated as word boundaries (classic scale-space heuristic).
    """
    h = ink_line.shape[0]
    proj = ink_line.sum(axis=0)
    on = proj > 0
    # collect runs of ink columns
    runs: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(on)))
    if not runs:
        return []
    gap_thr = max(h * gap_factor, 6)
    words: List[Tuple[int, int]] = [list(runs[0])]  # type: ignore[list-item]
    for (s, e) in runs[1:]:
        if s - words[-1][1] < gap_thr:
            words[-1][1] = e  # type: ignore[index]
        else:
            words.append([s, e])  # type: ignore[arg-type]
    return [(int(s), int(e)) for s, e in words]


def preprocess_upload(img: Image.Image) -> dict:
    """Full pipeline for one uploaded sample."""
    gray = to_grayscale(img)
    ink = binarize_ink(gray)
    gray, ink, angle = deskew(gray, ink)
    canvas = make_style_canvas(gray, ink)
    return {
        "gray": gray,
        "ink": ink,
        "skew_corrected_deg": angle,
        "style_canvas": canvas,   # PIL 768x768 'L'
        "line_bands": segment_lines(ink),
    }
