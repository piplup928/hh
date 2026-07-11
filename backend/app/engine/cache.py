"""Generation cache: memory LRU + disk persistence.

Key = (engine, style_id, normalized text, seed, quality, marks-signature).
Real-time typing hits this constantly — retyping a word, reflowing a
paragraph, or toggling formatting must not re-run diffusion.
"""
from __future__ import annotations

import hashlib
import io
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from PIL import Image

from ..config import CACHE_DIR, CACHE_MAX_ITEMS
from .base import InkResult, LineBox, WordBox


def make_key(engine: str, style_id: str, text: str, seed: Optional[int],
             quality: str, marks: Optional[dict] = None) -> str:
    payload = json.dumps(
        [engine, style_id, text, seed, quality, marks or {}],
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class InkCache:
    def __init__(self, root: Path = CACHE_DIR, max_items: int = CACHE_MAX_ITEMS):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items
        self._mem: OrderedDict[str, InkResult] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[InkResult]:
        with self._lock:
            if key in self._mem:
                self._mem.move_to_end(key)
                return self._mem[key]
        return self._load_disk(key)

    def put(self, key: str, result: InkResult) -> None:
        with self._lock:
            self._mem[key] = result
            self._mem.move_to_end(key)
            while len(self._mem) > self.max_items:
                self._mem.popitem(last=False)
        self._save_disk(key, result)

    # ------------------------------------------------------------------- disk
    def _paths(self, key: str):
        return self.root / f"{key}.png", self.root / f"{key}.json"

    def _save_disk(self, key: str, r: InkResult) -> None:
        png, meta = self._paths(key)
        try:
            r.image.save(png)
            meta.write_text(json.dumps({
                "engine": r.engine, "text": r.text, "xHeight": r.x_height,
                "seed": r.seed, "svg": r.svg, "meta": r.meta,
                "lines": [
                    {"text": ln.text, "x": ln.x, "y": ln.y, "w": ln.w, "h": ln.h,
                     "baseline": ln.baseline,
                     "words": [vars(w) for w in ln.words]}
                    for ln in r.lines],
            }))
        except OSError:
            pass  # cache is best-effort

    def _load_disk(self, key: str) -> Optional[InkResult]:
        png, meta = self._paths(key)
        if not (png.exists() and meta.exists()):
            return None
        try:
            data = json.loads(meta.read_text())
            img = Image.open(png).convert("RGBA")
            img.load()
            lines = [
                LineBox(text=ln["text"], x=ln["x"], y=ln["y"], w=ln["w"],
                        h=ln["h"], baseline=ln["baseline"],
                        words=[WordBox(**w) for w in ln.get("words", [])])
                for ln in data.get("lines", [])
            ]
            result = InkResult(
                engine=data["engine"], text=data["text"], image=img,
                lines=lines, x_height=data.get("xHeight", 0.0),
                seed=data.get("seed"), svg=data.get("svg"),
                meta=data.get("meta", {}))
            with self._lock:
                self._mem[key] = result
                self._mem.move_to_end(key)
            return result
        except (OSError, KeyError, ValueError):
            return None


cache = InkCache()
