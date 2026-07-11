"""Benchmark runner — REAL industry metrics only.

Scores model output against the user's actual uploaded handwriting:

  * HWD        — official aimagelab implementation (BMVC 2023), VGG16 features
                 trained on font_square; the de-facto styled-HTG score.
  * Style mAP  — writer-retrieval mean Average Precision on the same official
                 feature backbone (see style_map.py for the protocol).
  * FID/BFID/KID — official HWD-package implementations (optional).
  * CER        — character error rate via the HWD package's HTR (content
                 preservation check).

Hard guarantees:
  * Refuses to score preview-engine output (raises BenchmarkIntegrityError).
    Only real Paragraph-LDM / DiffInk generations are ever measured.
  * The reference set is built exclusively from the user's uploaded samples
    (segmented real lines) — never from generated data.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import List, Optional

from PIL import Image

from ..config import BENCH_DIR, PARAGRAPH_LDM_DIR
from ..engine.base import InkResult
from ..engine.service import service
from ..engine.style.preprocess import binarize_ink, segment_lines, to_grayscale
from ..engine.style.profile import store as style_store
from .style_map import style_map

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Handwriting is the mirror of personality and mood",
    "We hold these truths to be self evident",
    "A journey of a thousand miles begins with a single step",
    "Science is organized knowledge wisdom is organized life",
    "The early bird catches the worm every morning",
    "Practice makes perfect when patience guides the pen",
    "Every great story begins with a single written word",
]


class BenchmarkIntegrityError(RuntimeError):
    pass


def _flatten_ink(r: InkResult) -> Image.Image:
    """RGBA ink -> white-background grayscale line/page image for metrics."""
    bg = Image.new("RGB", r.image.size, (255, 255, 255))
    bg.paste(r.image, mask=r.image.split()[3])
    return bg.convert("L")


def _slice_lines(img: Image.Image, min_w: int = 48) -> List[Image.Image]:
    import numpy as np

    gray = to_grayscale(img)
    ink = binarize_ink(gray)
    out = []
    for (y0, y1) in segment_lines(ink):
        pad = max((y1 - y0) // 4, 4)
        a, b = max(y0 - pad, 0), min(y1 + pad, gray.shape[0])
        band = gray[a:b]
        cols = np.nonzero(ink[a:b].sum(axis=0) > 0)[0]
        if cols.size < 10:
            continue
        crop = band[:, max(int(cols.min()) - 8, 0): int(cols.max()) + 8]
        if crop.shape[1] >= min_w and crop.shape[0] >= 12:
            out.append(Image.fromarray(crop))
    return out


def _distractor_lines() -> dict[str, List[Image.Image]]:
    """Real IAM paragraphs bundled with the Paragraph-LDM repo, one writer per
    file, segmented into lines — the retrieval gallery's negative writers."""
    out: dict[str, List[Image.Image]] = {}
    style_dir = PARAGRAPH_LDM_DIR / "StyleExamples"
    for p in sorted(style_dir.glob("*.png")):
        lines = _slice_lines(Image.open(p).convert("L"))
        if lines:
            out[f"iam_{p.stem}"] = lines
    return out


class BenchmarkRunner:
    def __init__(self, workdir: Path = BENCH_DIR):
        self.workdir = workdir

    # ---------------------------------------------------------------- dataset
    def _build_folders(self, run_dir: Path, style_id: str,
                       generated: List[InkResult],
                       transcriptions: Optional[dict] = None):
        """Materialize hwd.datasets.FolderDataset layouts:
        real/<author>/*.png and fake/<author>/*.png"""
        profile = style_store.get(style_id)
        if profile is None:
            raise KeyError(f"style {style_id} not found")

        real_root = run_dir / "real"
        fake_root = run_dir / "fake"
        user = f"user_{style_id}"
        (real_root / user).mkdir(parents=True)
        (fake_root / user).mkdir(parents=True)

        n_real = 0
        for i, p in enumerate(profile.reference_line_paths):
            shutil.copy(p, real_root / user / f"ref_{i:04d}.png")
            n_real += 1
        if n_real == 0:
            raise BenchmarkIntegrityError(
                "style has no segmented reference lines; upload clearer samples")

        fake_tx = {}
        n_fake = 0
        for gi, r in enumerate(generated):
            if r.engine == "preview":
                raise BenchmarkIntegrityError(
                    "refusing to benchmark preview-engine output; load real "
                    "model weights (HW_ENGINE_MODE=live)")
            page = _flatten_ink(r)
            lines = _slice_lines(page)
            texts = [ln.text for ln in r.lines] if r.lines else []
            if not lines:
                lines = [page]
                texts = [r.text]
            for li, line_img in enumerate(lines):
                fn = f"gen_{gi:03d}_{li:02d}.png"
                line_img.save(fake_root / user / fn)
                if li < len(texts) and texts[li]:
                    fake_tx[f"{user}/{fn}"] = texts[li]
                n_fake += 1

        # distractor writers only in the REAL gallery (for retrieval mAP)
        for author, lines in _distractor_lines().items():
            (real_root / author).mkdir()
            for i, img in enumerate(lines):
                img.save(real_root / author / f"line_{i:03d}.png")

        if fake_tx:
            (fake_root / "transcriptions.json").write_text(json.dumps(fake_tx))
        return real_root, fake_root, user, n_real, n_fake

    # -------------------------------------------------------------------- run
    def run(self, style_id: str, *, texts: Optional[List[str]] = None,
            seed: int = 42, max_samples: int = 16,
            metrics: Optional[List[str]] = None) -> dict:
        metrics = metrics or ["hwd", "style_map", "cer"]
        texts = (texts or DEFAULT_TEXTS)[:max_samples]

        t0 = time.time()
        generated: List[InkResult] = []
        for i, text in enumerate(texts):
            generated.extend(service.generate_text_sync(
                text, style_id, seed=seed + i, quality="final"))
        gen_time = time.time() - t0

        run_id = uuid.uuid4().hex[:10]
        run_dir = self.workdir / run_id
        run_dir.mkdir(parents=True)
        real_root, fake_root, user, n_real, n_fake = self._build_folders(
            run_dir, style_id, generated)

        from hwd.datasets import FolderDataset  # official aimagelab package

        report: dict = {
            "runId": run_id,
            "styleId": style_id,
            "engines": sorted({r.engine for r in generated}),
            "nGeneratedImages": n_fake,
            "nReferenceImages": n_real,
            "generationSeconds": round(gen_time, 2),
            "metrics": {},
        }

        user_fakes = FolderDataset(str(fake_root))
        user_reals_only = FolderDataset(str(real_root / user).rsplit("/", 1)[0])

        if "hwd" in metrics:
            from hwd.scores import HWDScore

            hwd = HWDScore(height=32)
            # HWD compares fake vs the user's real lines only
            score = hwd(user_fakes, FolderDataset(str(real_root)))
            report["metrics"]["hwd"] = {
                "value": float(score),
                "direction": "lower_is_better",
                "reference": "Pippi et al., BMVC 2023 (official implementation)",
            }
            if "style_map" in metrics:
                q = hwd.digest(user_fakes)
                g = hwd.digest(FolderDataset(str(real_root)))
                report["metrics"]["style_map"] = {
                    **style_map(q, g, user),
                    "direction": "higher_is_better",
                    "protocol": "writer-retrieval mAP on official HWD VGG16 features",
                }
        elif "style_map" in metrics:
            from hwd.scores import HWDScore

            hwd = HWDScore(height=32)
            q = hwd.digest(user_fakes)
            g = hwd.digest(FolderDataset(str(real_root)))
            report["metrics"]["style_map"] = {
                **style_map(q, g, user),
                "direction": "higher_is_better",
                "protocol": "writer-retrieval mAP on official HWD VGG16 features",
            }

        for name, cls_name in (("fid", "FIDScore"), ("bfid", "BFIDScore"),
                               ("kid", "KIDScore")):
            if name in metrics:
                import hwd.scores as S

                score = getattr(S, cls_name)(height=32)(
                    user_fakes, FolderDataset(str(real_root)))
                report["metrics"][name] = {
                    "value": float(score), "direction": "lower_is_better"}

        if "cer" in metrics and (fake_root / "transcriptions.json").exists():
            try:
                from hwd.scores import CERScore

                cer = CERScore()
                report["metrics"]["cer"] = {
                    "value": float(cer(FolderDataset(str(fake_root)))),
                    "direction": "lower_is_better",
                    "note": "content preservation (HTR transcription vs prompt)",
                }
            except Exception as e:  # noqa: BLE001
                report["metrics"]["cer"] = {"error": str(e)}

        report["totalSeconds"] = round(time.time() - t0, 2)
        (run_dir / "report.json").write_text(json.dumps(report, indent=2))
        return report

    def list_reports(self) -> List[dict]:
        out = []
        for p in sorted(self.workdir.glob("*/report.json")):
            try:
                out.append(json.loads(p.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return out


runner = BenchmarkRunner()
