"""GenerationService — the pipeline heart.

typed text -> router (text|math runs) -> engine (Paragraph-LDM | DiffInk)
           -> cache -> style-preserving mark transforms -> InkResult payloads

Blocking diffusion calls run in a dedicated executor so the FastAPI event
loop (REST + WebSocket live typing) stays responsive; per-engine locks keep
GPU sampling serialized.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from ..config import ENGINE_MODE
from .base import EngineUnavailable, InkResult
from .cache import cache, make_key
from .diffink import engine as diffink_engine
from .ink_ops import apply_marks
from .paragraph_ldm import engine as ldm_engine
from .preview import engine as preview_engine
from .router import Run, latex_normalize, segment
from .style.profile import store as style_store
from .text_layout import chunk_text

log = logging.getLogger("engine.service")
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inkgen")


class GenerationService:
    def __init__(self):
        self.ldm = ldm_engine
        self.diffink = diffink_engine
        self.preview = preview_engine
        self.styles = style_store
        # in-flight de-dup: identical concurrent requests (editor redraws,
        # multiple ws connections) share one diffusion run
        self._inflight: dict = {}
        self._inflight_lock = __import__("threading").Lock()

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        return {
            "mode": ENGINE_MODE,
            "paragraphLdm": self.ldm.status(),
            "diffink": self.diffink.status(),
            "preview": self.preview.status(),
        }

    def _pick(self, kind: str):
        """Choose the real engine for a run kind; fall back to the clearly
        labelled preview engine only in preview mode."""
        primary = self.diffink if kind == "math" else self.ldm
        if primary.is_ready():
            return primary
        if ENGINE_MODE == "preview":
            return self.preview
        raise EngineUnavailable(
            f"{primary.name} is not ready and HW_ENGINE_MODE=live "
            f"(status: {primary.status()})")

    # -------------------------------------------------------------- generation
    def generate_run_sync(self, run: Run, style_id: str, *,
                          seed: Optional[int] = None, quality: str = "live",
                          marks: Optional[dict] = None) -> InkResult:
        import threading

        content = latex_normalize(run.content) if run.kind == "math" else run.content
        engine = self._pick(run.kind)
        base_key = make_key(engine.name, style_id, content, seed, quality, None)
        result = cache.get(base_key)
        if result is None:
            with self._inflight_lock:
                ev = self._inflight.get(base_key)
                owner = ev is None
                if owner:
                    ev = threading.Event()
                    self._inflight[base_key] = ev
            if owner:
                try:
                    result = engine.generate(content, style_id, seed=seed,
                                             quality=quality)
                    cache.put(base_key, result)
                finally:
                    with self._inflight_lock:
                        self._inflight.pop(base_key, None)
                    ev.set()
            else:
                # another worker is generating this exact content — wait for it
                ev.wait(timeout=1800)
                result = cache.get(base_key)
                if result is None:
                    result = engine.generate(content, style_id, seed=seed,
                                             quality=quality)
                    cache.put(base_key, result)
        if marks:
            marked_key = make_key(engine.name, style_id, content, seed, quality, marks)
            marked = cache.get(marked_key)
            if marked is None:
                profile = self.styles.get(style_id)
                sw = profile.metrics.stroke_width_px if profile else 2.0
                img = apply_marks(
                    result.image, profile_stroke_width=sw,
                    bold=bool(marks.get("bold")), italic=bool(marks.get("italic")),
                    color=marks.get("color"),
                    size_factor=float(marks.get("sizeFactor", 1.0)))
                marked = InkResult(
                    engine=result.engine, text=result.text, image=img,
                    lines=result.lines, x_height=result.x_height,
                    svg=result.svg, seed=result.seed,
                    meta={**result.meta, "marks": marks})
                cache.put(marked_key, marked)
            result = marked
        return result

    async def generate_run(self, run: Run, style_id: str, **kw) -> InkResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, lambda: self.generate_run_sync(run, style_id, **kw))

    def generate_text_sync(self, text: str, style_id: str, *,
                           seed: Optional[int] = None, quality: str = "live",
                           marks: Optional[dict] = None) -> List[InkResult]:
        """Segment arbitrary editor text and generate every run.

        Long prose runs are chunked to the LDM page capacity so paragraphs of
        any length work; each chunk is one diffusion call (cached)."""
        results: List[InkResult] = []
        for run in segment(text):
            if run.kind == "text":
                from ..config import LDM_MAX_CHARS_PER_LINE, LDM_MAX_LINES

                for chunk in chunk_text(run.content, LDM_MAX_CHARS_PER_LINE, LDM_MAX_LINES):
                    results.append(self.generate_run_sync(
                        Run("text", chunk), style_id, seed=seed,
                        quality=quality, marks=marks))
            else:
                results.append(self.generate_run_sync(
                    run, style_id, seed=seed, quality=quality, marks=marks))
        return results

    async def generate_text(self, text: str, style_id: str, **kw) -> List[InkResult]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, lambda: self.generate_text_sync(text, style_id, **kw))


service = GenerationService()
