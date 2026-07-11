"""Persisted per-user handwriting style profiles.

A StyleProfile bundles everything the platform learns from the uploaded
samples:

  * the 768x768 style canvases fed to Paragraph-LDM's style encoder (the
    actual zero-shot conditioning — the model imitates directly from these)
  * interpretable StyleMetrics driving layout + DiffInk pen model
  * segmented reference line/word crops used as the REAL reference set by the
    benchmark runner (HWD / style-mAP compare generated ink against these)
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from ...config import STYLES_DIR
from .analyzer import StyleMetrics, analyze, merge_metrics
from .preprocess import preprocess_upload, segment_words


@dataclass
class StyleProfile:
    style_id: str
    name: str
    created_at: float
    n_samples: int
    metrics: StyleMetrics
    dir: Path

    @property
    def canvas_paths(self) -> List[Path]:
        return sorted((self.dir / "canvases").glob("*.png"))

    @property
    def reference_line_paths(self) -> List[Path]:
        return sorted((self.dir / "reference_lines").glob("*.png"))

    def to_dict(self) -> dict:
        return {
            "styleId": self.style_id,
            "name": self.name,
            "createdAt": self.created_at,
            "nSamples": self.n_samples,
            "metrics": self.metrics.to_dict(),
            "nReferenceLines": len(self.reference_line_paths),
        }


class StyleStore:
    """Filesystem-backed style registry."""

    def __init__(self, root: Path = STYLES_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ create
    def create(self, images: List[Image.Image], name: str = "My handwriting") -> StyleProfile:
        style_id = uuid.uuid4().hex[:12]
        sdir = self.root / style_id
        (sdir / "canvases").mkdir(parents=True)
        (sdir / "reference_lines").mkdir()
        (sdir / "uploads").mkdir()

        all_metrics = []
        line_idx = 0
        for i, img in enumerate(images):
            img.convert("RGB").save(sdir / "uploads" / f"upload_{i:02d}.png")
            pre = preprocess_upload(img)
            pre["style_canvas"].save(sdir / "canvases" / f"canvas_{i:02d}.png")
            m = analyze(pre["gray"], pre["ink"])
            all_metrics.append(m)
            # export per-line reference crops (real handwriting ground truth
            # for benchmarking + retrieval galleries)
            gray, ink = pre["gray"], pre["ink"]
            for (y0, y1) in pre["line_bands"]:
                pad = max((y1 - y0) // 4, 4)
                a, b = max(y0 - pad, 0), min(y1 + pad, gray.shape[0])
                band = gray[a:b]
                cols = np.nonzero(ink[a:b].sum(axis=0) > 0)[0]
                if cols.size < 20 or (b - a) < 12:
                    continue
                crop = band[:, max(cols.min() - 8, 0): cols.max() + 8]
                Image.fromarray(crop).save(
                    sdir / "reference_lines" / f"line_{line_idx:04d}.png")
                line_idx += 1

        metrics = merge_metrics(all_metrics)
        profile = StyleProfile(
            style_id=style_id, name=name, created_at=time.time(),
            n_samples=len(images), metrics=metrics, dir=sdir,
        )
        (sdir / "profile.json").write_text(json.dumps({
            "styleId": style_id, "name": name, "createdAt": profile.created_at,
            "nSamples": len(images), "metrics": metrics.to_dict(),
        }, indent=2))
        return profile

    # -------------------------------------------------------------------- load
    def get(self, style_id: str) -> Optional[StyleProfile]:
        sdir = self.root / style_id
        pj = sdir / "profile.json"
        if not pj.exists():
            return None
        data = json.loads(pj.read_text())
        return StyleProfile(
            style_id=data["styleId"], name=data["name"],
            created_at=data["createdAt"], n_samples=data["nSamples"],
            metrics=StyleMetrics(**data["metrics"]), dir=sdir,
        )

    def list(self) -> List[StyleProfile]:
        out = []
        for pj in sorted(self.root.glob("*/profile.json")):
            p = self.get(pj.parent.name)
            if p:
                out.append(p)
        return out

    def delete(self, style_id: str) -> bool:
        sdir = self.root / style_id
        if sdir.exists():
            shutil.rmtree(sdir)
            return True
        return False

    def pick_style_canvas(self, profile: StyleProfile, seed: Optional[int] = None) -> Image.Image:
        paths = profile.canvas_paths
        if not paths:
            raise FileNotFoundError(f"style {profile.style_id} has no canvases")
        rng = np.random.default_rng(seed)
        return Image.open(paths[int(rng.integers(0, len(paths)))]).convert("L")


store = StyleStore()
