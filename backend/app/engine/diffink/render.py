"""Trajectory rendering: (dx, dy, pen) sequences -> styled SVG + RGBA ink.

The pen model applies the user's StyleProfile: stroke width (with pressure
jitter derived from stroke_width_std), slant shear, and darkness — so DiffInk
output composes seamlessly with Paragraph-LDM ink on the page.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

Stroke = List[Tuple[float, float]]


def offsets_to_strokes(seq: np.ndarray, pen_threshold: float = 0.5) -> List[Stroke]:
    """(T,3) array of (dx, dy, pen) -> list of absolute-coordinate strokes."""
    strokes: List[Stroke] = []
    cur: Stroke = []
    x = y = 0.0
    for dx, dy, pen in seq:
        x, y = x + float(dx), y + float(dy)
        if pen >= pen_threshold:
            cur.append((x, y))
        else:
            if len(cur) > 1:
                strokes.append(cur)
            cur = []
    if len(cur) > 1:
        strokes.append(cur)
    return strokes


def normalize_strokes(strokes: List[Stroke], target_height: float) -> List[Stroke]:
    pts = np.array([p for s in strokes for p in s], dtype=np.float64)
    if pts.size == 0:
        return strokes
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    h = max(mx[1] - mn[1], 1e-6)
    scale = target_height / h
    return [[((px - mn[0]) * scale, (py - mn[1]) * scale) for px, py in s] for s in strokes]


def shear_strokes(strokes: List[Stroke], slant_deg: float, height: float) -> List[Stroke]:
    if abs(slant_deg) < 0.5:
        return strokes
    k = math.tan(math.radians(slant_deg))
    return [[(px + (height - py) * k, py) for px, py in s] for s in strokes]


def smooth_stroke(stroke: Stroke, iterations: int = 1) -> Stroke:
    """Chaikin corner cutting for natural pen curvature."""
    pts = stroke
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p, q = np.array(pts[i]), np.array(pts[i + 1])
            out.append(tuple(0.75 * p + 0.25 * q))
            out.append(tuple(0.25 * p + 0.75 * q))
        out.append(pts[-1])
        pts = out
    return pts


def strokes_to_svg(strokes: List[Stroke], stroke_width: float, color: str = "#000",
                   pad: int = 8) -> Tuple[str, Tuple[int, int]]:
    pts = np.array([p for s in strokes for p in s], dtype=np.float64)
    if pts.size == 0:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'/>", (1, 1)
    mx = pts.max(axis=0)
    w, h = int(mx[0]) + 2 * pad, int(mx[1]) + 2 * pad
    paths = []
    for s in strokes:
        d = f"M {s[0][0]+pad:.1f} {s[0][1]+pad:.1f} " + " ".join(
            f"L {x+pad:.1f} {y+pad:.1f}" for x, y in s[1:])
        paths.append(
            f"<path d='{d}' fill='none' stroke='{color}' "
            f"stroke-width='{stroke_width:.2f}' stroke-linecap='round' "
            f"stroke-linejoin='round'/>")
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' "
           f"viewBox='0 0 {w} {h}'>" + "".join(paths) + "</svg>")
    return svg, (w, h)


def rasterize_strokes(strokes: List[Stroke], stroke_width: float,
                      width_jitter: float = 0.0, darkness: float = 0.9,
                      pad: int = 8, supersample: int = 3) -> Image.Image:
    """Render strokes to an RGBA ink raster (black ink in alpha channel)."""
    pts = np.array([p for s in strokes for p in s], dtype=np.float64)
    if pts.size == 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    mx = pts.max(axis=0)
    W, H = int(mx[0]) + 2 * pad, int(mx[1]) + 2 * pad
    ss = supersample
    img = Image.new("L", (W * ss, H * ss), 0)
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(0)
    for s in strokes:
        s = smooth_stroke(s, 2)
        for i in range(len(s) - 1):
            w_i = stroke_width
            if width_jitter > 0:
                w_i = max(stroke_width + rng.normal(0, width_jitter * 0.35), 0.6)
            x0, y0 = s[i]
            x1, y1 = s[i + 1]
            draw.line(
                [(x0 + pad) * ss, (y0 + pad) * ss, (x1 + pad) * ss, (y1 + pad) * ss],
                fill=255, width=max(int(round(w_i * ss)), 1))
            r = w_i * ss / 2
            draw.ellipse([(x1 + pad) * ss - r, (y1 + pad) * ss - r,
                          (x1 + pad) * ss + r, (y1 + pad) * ss + r], fill=255)
    img = img.resize((W, H), Image.LANCZOS)
    alpha = (np.array(img, dtype=np.float32) / 255.0 * darkness * 255).astype(np.uint8)
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, "RGBA")
