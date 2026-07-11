"""Style-preserving rich-text transforms on generated ink.

The generative models own the *style*; formatting is applied as image-space
operations on the RGBA ink so bold/italic/color/size never break the learned
handwriting:

  bold      -> morphological dilation scaled to the profile's stroke width
               (reads as pressing harder with the same pen)
  italic    -> additional shear on top of the writer's natural slant
  underline -> synthesized pen stroke with baseline-drift jitter matching the
               profile (so it looks drawn, not ruled)
  color / highlight -> recolor via the alpha channel (ink texture kept)
  size / spacing    -> geometric ops on sprites; layout handled by the editor
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def _split(img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array(img.convert("RGBA"))
    return arr[..., :3], arr[..., 3]


def recolor(img: Image.Image, hex_color: str) -> Image.Image:
    """Apply ink color while keeping the generated stroke texture (alpha)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    _, a = _split(img)
    out = np.zeros((*a.shape, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = r, g, b, a
    return Image.fromarray(out, "RGBA")


def embolden(img: Image.Image, stroke_width_px: float, strength: float = 1.0) -> Image.Image:
    """Thicken strokes ~35% per strength unit via distance-scaled dilation."""
    rgb, a = _split(img)
    grow = max(int(round(stroke_width_px * 0.35 * strength)), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    a2 = cv2.dilate(a, kernel)
    # soften so it still looks like wetter ink, not a stamp
    a2 = cv2.GaussianBlur(a2, (3, 3), 0)
    a2 = np.maximum(a, a2 * 0.92).astype(np.uint8)
    out = np.dstack([rgb, a2])
    return Image.fromarray(out, "RGBA")


def italicize(img: Image.Image, extra_slant_deg: float = 9.0) -> Image.Image:
    """Shear the ink further right; keeps stroke texture."""
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[:2]
    k = math.tan(math.radians(extra_slant_deg))
    pad = int(abs(k) * h) + 1
    M = np.float32([[1, k, pad if k < 0 else 0], [0, 1, 0]])
    out = cv2.warpAffine(arr, M, (w + pad, h), flags=cv2.INTER_LINEAR,
                         borderValue=(0, 0, 0, 0))
    return Image.fromarray(out, "RGBA")


def scale(img: Image.Image, factor: float) -> Image.Image:
    if abs(factor - 1.0) < 1e-3:
        return img
    w = max(int(img.width * factor), 1)
    h = max(int(img.height * factor), 1)
    return img.resize((w, h), Image.LANCZOS)


def underline_stroke(width: int, stroke_width_px: float, drift_px: float,
                     color: str = "#000000", seed: int = 0) -> Image.Image:
    """A hand-drawn-looking underline segment matching the pen profile."""
    h = max(int(stroke_width_px * 3 + drift_px * 2), 6)
    img = Image.new("RGBA", (max(width, 4), h), (0, 0, 0, 0))
    rng = np.random.default_rng(seed)
    n = max(width // 12, 4)
    xs = np.linspace(2, width - 2, n)
    ys = h / 2 + np.cumsum(rng.normal(0, drift_px * 0.25, n))
    ys -= ys.mean() - h / 2
    pts = np.stack([xs, np.clip(ys, 1, h - 2)], axis=1)
    arr = np.array(img)
    for i in range(len(pts) - 1):
        cv2.line(arr, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)),
                 (0, 0, 0, 255), max(int(stroke_width_px), 1), cv2.LINE_AA)
    out = Image.fromarray(arr, "RGBA")
    return recolor(out, color) if color != "#000000" else out


def apply_marks(img: Image.Image, *, profile_stroke_width: float = 2.0,
                bold: bool = False, italic: bool = False,
                color: Optional[str] = None, size_factor: float = 1.0) -> Image.Image:
    """Compose the standard mark set in a stable order."""
    out = img
    if bold:
        out = embolden(out, profile_stroke_width)
    if italic:
        out = italicize(out)
    if color:
        out = recolor(out, color)
    if size_factor != 1.0:
        out = scale(out, size_factor)
    return out
