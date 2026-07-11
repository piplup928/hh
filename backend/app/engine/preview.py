"""Preview engine — development fallback ONLY.

When HW_ENGINE_MODE=preview and a real model's weights are missing, this
engine renders a cursive-font approximation so the editor remains usable.
Everything it produces is tagged engine="preview"; the benchmark runner
refuses preview output, and live mode never touches this module.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import InkResult, LineBox, WordBox
from .style.profile import StyleStore, store as style_store
from .text_layout import wrap_paragraph

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
]


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)  # type: ignore[arg-type]


class PreviewEngine:
    name = "preview"

    def __init__(self, styles: StyleStore = style_store):
        self.styles = styles

    def is_ready(self) -> bool:
        return True

    def status(self) -> dict:
        return {"engine": self.name, "loaded": True,
                "note": "font-based development preview; never used in benchmarks"}

    def generate(self, text: str, style_id: str, *, seed: Optional[int] = None,
                 quality: str = "live") -> InkResult:
        profile = self.styles.get(style_id)
        m = profile.metrics if profile else None
        size = int((m.x_height_px if m else 24) * (m.ascender_ratio if m else 1.7))
        size = int(np.clip(size, 18, 64))
        font = _find_font(size)
        lines = wrap_paragraph(text, 48, 32) or [""]
        line_h = int((m.line_spacing_px if m else size * 1.6))
        rng = np.random.default_rng(seed if seed is not None else 0)

        widths = [font.getbbox(ln)[2] if ln else 1 for ln in lines]
        W = max(max(widths) + 32, 8)
        H = line_h * len(lines) + 24
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        boxes: List[LineBox] = []
        for i, ln in enumerate(lines):
            y = 8 + i * line_h + int(rng.normal(0, (m.baseline_drift_px if m else 1.5) * 0.6))
            draw.text((16, y), ln, font=font, fill=(0, 0, 0, 235))
            bb = font.getbbox(ln or " ")
            lb = LineBox(text=ln, x=16, y=y, w=bb[2], h=bb[3] - bb[1],
                         baseline=y + int(size * 0.85))
            x = 16
            for wtext in ln.split():
                ww = font.getbbox(wtext)[2]
                lb.words.append(WordBox(text=wtext, x=x, y=y, w=ww,
                                        h=bb[3] - bb[1], baseline=lb.baseline))
                x += ww + font.getbbox(" ")[2]
            boxes.append(lb)

        if m and abs(m.slant_deg) > 2:  # rough slant hint so preview ~ tracks style
            arr = np.array(img)
            import cv2
            k = math.tan(math.radians(np.clip(m.slant_deg, -25, 25) * 0.5))
            M = np.float32([[1, k, 0], [0, 1, 0]])
            arr = cv2.warpAffine(arr, M, (arr.shape[1] + int(abs(k) * arr.shape[0]) + 1,
                                          arr.shape[0]), borderValue=(0, 0, 0, 0))
            img = Image.fromarray(arr, "RGBA")

        return InkResult(engine="preview", text=text, image=img, lines=boxes,
                         x_height=float(m.x_height_px if m else size * 0.6),
                         seed=seed, meta={"preview": True})


engine = PreviewEngine()
