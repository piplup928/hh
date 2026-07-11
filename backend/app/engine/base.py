"""Shared engine contracts.

Every generator (Paragraph-LDM for prose, DiffInk for math, preview fallback)
produces `InkResult` objects: a transparent-background RGBA ink raster plus
layout metadata, so the rest of the pipeline (rich-text transforms, renderer,
editor overlay, benchmark runner) is engine-agnostic.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Protocol

from PIL import Image

EngineName = Literal["paragraph_ldm", "diffink", "preview"]


@dataclass
class WordBox:
    """Bounding box of one word inside the ink raster (pixels)."""

    text: str
    x: int
    y: int
    w: int
    h: int
    baseline: int  # y of the baseline within the raster


@dataclass
class LineBox:
    text: str
    x: int
    y: int
    w: int
    h: int
    baseline: int
    words: List[WordBox] = field(default_factory=list)


@dataclass
class InkResult:
    """A generated piece of handwriting."""

    engine: EngineName
    text: str                      # source text (or LaTeX for math)
    image: Image.Image             # RGBA, ink in alpha channel, black ink
    lines: List[LineBox] = field(default_factory=list)
    x_height: float = 0.0          # estimated x-height in px (for scaling to editor font size)
    svg: Optional[str] = None      # vector strokes when the engine is trajectory-based (DiffInk)
    seed: Optional[int] = None
    meta: dict = field(default_factory=dict)

    def png_base64(self) -> str:
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def to_payload(self) -> dict:
        return {
            "engine": self.engine,
            "text": self.text,
            "png": self.png_base64(),
            "svg": self.svg,
            "width": self.image.width,
            "height": self.image.height,
            "xHeight": self.x_height,
            "seed": self.seed,
            "lines": [
                {
                    "text": ln.text,
                    "x": ln.x, "y": ln.y, "w": ln.w, "h": ln.h,
                    "baseline": ln.baseline,
                    "words": [
                        {"text": w.text, "x": w.x, "y": w.y, "w": w.w, "h": w.h,
                         "baseline": w.baseline}
                        for w in ln.words
                    ],
                }
                for ln in self.lines
            ],
            "meta": self.meta,
        }


class HandwritingEngine(Protocol):
    """Contract implemented by every generator."""

    name: EngineName

    def is_ready(self) -> bool:
        """True when real weights are loaded and live generation is possible."""
        ...

    def generate(self, text: str, style_id: str, *, seed: Optional[int] = None,
                 quality: str = "live") -> InkResult:
        ...


class EngineUnavailable(RuntimeError):
    """Raised in live mode when a model's weights are missing/not loaded."""
