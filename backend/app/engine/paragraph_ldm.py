"""Paragraph-LDM engine.

Wraps the official implementation of
"Zero-Shot Paragraph-level Handwriting Imitation with Latent Diffusion Models"
(Mayr et al., arXiv:2409.00786 — vendored at backend/vendor/paragraph_handwriting_imitation_ldm)
behind the platform's HandwritingEngine contract.

Design notes
------------
* The vendor demo (`demo.py` -> ConditionalSampler.start_sampling) is CUDA-
  hardcoded and driven by global `Parameters.py`. We import the vendor modules
  directly (LatentDiffusion, DDIMSampler, Alphabet) and rebuild the
  conditioning tensors device-agnostically, mirroring
  `make_conditioning(use_conditioning == CROSS_CONDITIONING)`.
* Conditioning = (text logits, causal mask, key padding, style image,
  style padding). The style image is the user's preprocessed 768x768 sample —
  this is the zero-shot imitation path: NO fine-tuning per user, the style
  encoder consumes the raw sample.
* Classifier-free guidance uses an empty string + blank style as the
  unconditional branch, exactly like the vendor sampler.
* Output post-processing converts the sampled page to a transparent ink RGBA
  raster and extracts line/word boxes so the editor can flow the ink.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import List, Optional

import numpy as np
from PIL import Image

from ..config import (
    LDM_CANVAS,
    LDM_GUIDANCE,
    LDM_MAX_CHARS_PER_LINE,
    LDM_MAX_LINES,
    LDM_STEPS,
    LDM_STEPS_FINAL,
    PARAGRAPH_LDM_CKPT,
    PARAGRAPH_LDM_DIR,
    resolve_device,
)
from .base import EngineUnavailable, InkResult, LineBox, WordBox
from .style.preprocess import segment_lines, segment_words
from .style.profile import StyleStore, store as style_store
from .text_layout import wrap_paragraph

log = logging.getLogger("engine.paragraph_ldm")


def _import_vendor():
    """Make the vendored repo importable (it uses repo-root absolute imports)."""
    root = str(PARAGRAPH_LDM_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


class ParagraphLDMEngine:
    name = "paragraph_ldm"

    def __init__(self, ckpt_path=PARAGRAPH_LDM_CKPT, styles: StyleStore = style_store):
        self.ckpt_path = ckpt_path
        self.styles = styles
        self.device = resolve_device()
        self._model = None
        self._sampler = None
        self._alphabet = None
        self._lock = threading.Lock()  # GPU sampling is serialized
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ status
    def is_ready(self) -> bool:
        if self._model is not None:
            return True
        return self.ckpt_path.exists() and self._load_error is None

    def status(self) -> dict:
        return {
            "engine": self.name,
            "loaded": self._model is not None,
            "checkpoint": str(self.ckpt_path),
            "checkpointPresent": self.ckpt_path.exists(),
            "device": self.device,
            "error": self._load_error,
        }

    # ------------------------------------------------------------------- model
    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.ckpt_path.exists():
                raise EngineUnavailable(
                    f"Paragraph-LDM checkpoint missing at {self.ckpt_path}. "
                    "Run backend/scripts/setup_models.sh"
                )
            try:
                _import_vendor()
                import torch
                from omegaconf import OmegaConf

                from src.diffusion.ddim import DDIMSampler          # vendor
                from src.diffusion.ddpm import LatentDiffusion      # vendor
                from src.data.utils.alphabet import Alphabet        # vendor
                from src.utils.utils import instantiate_completely  # vendor

                cwd = os.getcwd()
                try:
                    # vendor config loader resolves relative to CWD
                    os.chdir(str(PARAGRAPH_LDM_DIR))
                    model = instantiate_completely(
                        os.path.join("Diffusion", "ldm"), "ours.yaml"
                    )
                finally:
                    os.chdir(cwd)

                sd = self._load_checkpoint_file(torch)
                state = sd.get("state_dict", sd)
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    log.warning("LDM load: %d missing keys (first: %s)",
                                len(missing), missing[:3])
                model.to(self.device)
                model.eval()
                self._model = model
                self._sampler = DDIMSampler(model)
                self._alphabet = Alphabet()
                log.info("Paragraph-LDM loaded on %s", self.device)
            except EngineUnavailable:
                raise
            except Exception as e:  # noqa: BLE001
                self._load_error = f"{type(e).__name__}: {e}"
                log.exception("Paragraph-LDM load failed")
                raise EngineUnavailable(self._load_error) from e

    # ------------------------------------------------------------- ckpt loading
    def _load_checkpoint_file(self, torch):
        """Load the official checkpoint, tolerating packaging differences.

        The Drive release has shipped in several shapes over time: a raw
        torch checkpoint, or a tar/zip ARCHIVE containing the .ckpt file.
        A wrapping archive surfaces as torch.load failing with e.g.
        KeyError: "filename 'storages' not found" (valid tar, but not the
        ancient torch tar format). We detect that, extract the inner
        checkpoint next to the original (cached for future startups), and
        load it. weights_only=False everywhere: trusted authors' release.
        """
        import tarfile
        import zipfile

        extracted = self.ckpt_path.with_name(self.ckpt_path.stem + "_extracted.ckpt")
        if extracted.exists():
            log.info("loading previously extracted checkpoint %s", extracted)
            return torch.load(str(extracted), map_location="cpu", weights_only=False)

        try:
            return torch.load(str(self.ckpt_path), map_location="cpu",
                              weights_only=False)
        except Exception as first_err:  # noqa: BLE001
            log.warning("direct torch.load failed (%s); checking whether the "
                        "file is an archive wrapping the checkpoint",
                        first_err)

        def pick(names):
            """Choose the most checkpoint-looking member name."""
            cands = [n for n in names
                     if n.lower().endswith((".ckpt", ".pt", ".pth", ".bin"))]
            if not cands:
                cands = list(names)
            for hint in ("ldm", "last", "diffusion", "epoch"):
                hinted = [n for n in cands if hint in n.lower()]
                if hinted:
                    cands = hinted
                    break
            return cands

        path = str(self.ckpt_path)
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as tf:
                members = {m.name: m for m in tf.getmembers() if m.isfile()}
                if not members:
                    raise RuntimeError("checkpoint tar archive is empty")
                names = pick(members.keys())
                best = max(names, key=lambda n: members[n].size)
                log.info("extracting %r (%.0f MB) from checkpoint tar (members: %s)",
                         best, members[best].size / 2**20, sorted(members)[:8])
                with tf.extractfile(members[best]) as src, open(extracted, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 24)
                        if not chunk:
                            break
                        dst.write(chunk)
            return torch.load(str(extracted), map_location="cpu", weights_only=False)

        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                infos = {i.filename: i for i in zf.infolist() if not i.is_dir()}
                # a real torch>=1.6 checkpoint is itself a zip containing
                # data.pkl — that case loads directly above, so reaching here
                # means it's a wrapper zip
                names = pick(infos.keys())
                best = max(names, key=lambda n: infos[n].file_size)
                log.info("extracting %r (%.0f MB) from checkpoint zip",
                         best, infos[best].file_size / 2**20)
                with zf.open(infos[best]) as src, open(extracted, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 24)
                        if not chunk:
                            break
                        dst.write(chunk)
            return torch.load(str(extracted), map_location="cpu", weights_only=False)

        # neither loadable nor an archive: probably a corrupt/HTML download
        head = open(path, "rb").read(256)
        raise RuntimeError(
            f"{self.ckpt_path} is not a torch checkpoint nor a tar/zip archive "
            f"(first bytes: {head[:60]!r}). The Google Drive download likely "
            "failed (quota/HTML page) — delete the file and re-run "
            "scripts/setup_models.sh")

    # ------------------------------------------------------------ conditioning
    def _subsequent_mask(self, size: int):
        import torch

        m = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
        out = torch.zeros(size, size)
        out.masked_fill_(m, float("-inf"))
        return out

    def _make_conditioning(self, text: str, style_img: Image.Image, blank_style: bool = False):
        """Mirror of vendor make_conditioning (CROSS_CONDITIONING path), device-agnostic."""
        import torch

        dev = self.device
        a = self._alphabet
        string_logits = a.string_to_logits(text).to(dev)
        n = len(text)

        logits = torch.ones((1, n + 2), device=dev)
        logits[0][0] = 2.0        # <SOS>
        logits[0][-1] = 3.0       # <EOS>
        if n:
            logits[0][1:-1] = string_logits
        logits = logits.long()

        tgt_mask = self._subsequent_mask(logits.shape[1] - 1).to(dev)
        key_padding = torch.eq(
            logits, torch.ones_like(logits)
        )  # PAD token == 1

        arr = np.array(style_img.convert("L"), dtype=np.float32)
        if arr.shape != (LDM_CANVAS, LDM_CANVAS):
            style_img = style_img.resize((LDM_CANVAS, LDM_CANVAS), Image.LANCZOS)
            arr = np.array(style_img.convert("L"), dtype=np.float32)
        style = (1.0 - torch.from_numpy(arr) / 255.0) - 0.5
        style = style.to(dev).unsqueeze(0).unsqueeze(0)  # (1,1,768,768)
        if blank_style:
            style = torch.zeros_like(style) - 0.5

        model = self._model
        if getattr(model, "cond_stage_concat_mode", False):
            style_pad = torch.zeros((1, 1), dtype=torch.bool, device=dev)
            key_padding = torch.cat((key_padding, style_pad), dim=1)

        con = (logits, tgt_mask, key_padding, style, None)
        c = model.get_learned_conditioning(con)
        pred_logits = torch.ones((1, n + 1), device=dev)
        pred_logits[0][-1] = 3.0
        if n:
            pred_logits[0][0:-1] = string_logits
        return c, pred_logits.long()

    # ---------------------------------------------------------------- sampling
    _SAFE_DDIM_STEPS = (2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500)

    @classmethod
    def _safe_steps(cls, steps: int) -> int:
        """The vendored CompVis DDIM scheduler indexes out of range when the
        step count doesn't divide the 1000 DDPM timesteps; snap to the nearest
        divisor."""
        if 1000 % max(steps, 1) == 0:
            return steps
        return min(cls._SAFE_DDIM_STEPS, key=lambda s: abs(s - steps))

    def _sample_page(self, text: str, style_img: Image.Image, steps: int,
                     guidance: float, seed: Optional[int]) -> Image.Image:
        steps = self._safe_steps(steps)
        import torch

        self._ensure_model()
        model, sampler = self._model, self._sampler
        with self._lock, torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            c, logits = self._make_conditioning(text, style_img)
            uc, _ = self._make_conditioning("", style_img, blank_style=True)
            shape = [
                model.model.diffusion_model.in_channels,
                model.model.diffusion_model.image_size[0],
                model.model.diffusion_model.image_size[1],
            ]
            samples, _ = sampler.sample(
                S=steps, conditioning=c, batch_size=1, shape=shape,
                verbose=False, quantize_x0=False,
                unconditional_guidance_scale=guidance,
                unconditional_conditioning=uc, eta=0.0, logits=logits,
            )
            x = model.decode_first_stage(samples)
            # vendor save path: img = clamp(1 - (x + 0.5), 0, 1); white bg, dark ink
            img = torch.clamp(1.0 - (x + 0.5), min=0.0, max=1.0)
            arr = (255.0 * img[0, 0].detach().cpu().numpy()).astype(np.uint8)
            return Image.fromarray(arr)

    # ---------------------------------------------------------- postprocessing
    @staticmethod
    def page_to_ink(page: Image.Image) -> Image.Image:
        """White-page grayscale -> RGBA with ink as alpha (black ink)."""
        g = np.array(page.convert("L"), dtype=np.float32)
        bg = np.percentile(g, 90)
        alpha = np.clip((bg - g) / max(bg, 1) * 1.6, 0.0, 1.0)
        alpha = (alpha * 255).astype(np.uint8)
        rgba = np.zeros((*g.shape, 4), np.uint8)
        rgba[..., 3] = alpha
        return Image.fromarray(rgba, "RGBA")

    @staticmethod
    def extract_layout(page: Image.Image, lines_text: List[str]) -> List[LineBox]:
        """Slice a sampled page into line and word boxes.

        Word slicing uses whitespace-valley segmentation; when the detected
        word count disagrees with the transcript, the line box is kept and
        word boxes are interpolated proportionally to text length (so the
        editor can still map editing positions to ink)."""
        g = np.array(page.convert("L"))
        ink = (g < np.percentile(g, 90) - 30).astype(np.uint8) * 255
        bands = segment_lines(ink)
        boxes: List[LineBox] = []
        for i, (y0, y1) in enumerate(bands[: len(lines_text)]):
            band = ink[y0:y1]
            cols = np.nonzero(band.sum(axis=0) > 0)[0]
            if cols.size == 0:
                continue
            x0, x1 = int(cols.min()), int(cols.max())
            text = lines_text[i] if i < len(lines_text) else ""
            baseline = y1 - max((y1 - y0) // 5, 2)
            lb = LineBox(text=text, x=x0, y=int(y0), w=x1 - x0, h=int(y1 - y0),
                         baseline=int(baseline))
            words = text.split()
            spans = segment_words(band)
            spans = [(s, e) for s, e in spans if e - s > 2]
            if words and len(spans) == len(words):
                for wtext, (s, e) in zip(words, spans):
                    seg = band[:, s:e]
                    rows = np.nonzero(seg.sum(axis=1) > 0)[0]
                    wy0 = int(y0 + (rows.min() if rows.size else 0))
                    wy1 = int(y0 + (rows.max() if rows.size else band.shape[0]))
                    lb.words.append(WordBox(text=wtext, x=int(s), y=wy0,
                                            w=int(e - s), h=wy1 - wy0,
                                            baseline=int(baseline)))
            elif words:
                # proportional fallback
                total = sum(len(w) for w in words) + (len(words) - 1)
                cx = x0
                width = x1 - x0
                for wtext in words:
                    frac = (len(wtext) + 0.5) / total
                    ww = int(width * frac)
                    lb.words.append(WordBox(text=wtext, x=cx, y=int(y0),
                                            w=ww, h=int(y1 - y0),
                                            baseline=int(baseline)))
                    cx += ww + int(width / total)
            boxes.append(lb)
        return boxes

    # -------------------------------------------------------------------- api
    def generate(self, text: str, style_id: str, *, seed: Optional[int] = None,
                 quality: str = "live") -> InkResult:
        profile = self.styles.get(style_id)
        if profile is None:
            raise KeyError(f"unknown style {style_id}")
        style_img = self.styles.pick_style_canvas(profile, seed=seed)

        lines = wrap_paragraph(text, LDM_MAX_CHARS_PER_LINE, LDM_MAX_LINES)
        cond_text = "\n".join(lines)
        steps = LDM_STEPS if quality == "live" else LDM_STEPS_FINAL
        # without CUDA, 50-step sampling takes minutes; default live typing to
        # 20 steps (still CFG-guided) unless the user pinned HW_LDM_STEPS
        if quality == "live" and self.device != "cuda" and "HW_LDM_STEPS" not in os.environ:
            steps = 20
        t0 = time.time()
        page = self._sample_page(cond_text, style_img, steps, LDM_GUIDANCE, seed)
        log.info("generated %r (%d chars) in %.1fs — steps=%d device=%s",
                 text[:40], len(text), time.time() - t0, steps, self.device)
        layout = self.extract_layout(page, lines)
        ink = self.page_to_ink(page)
        x_h = profile.metrics.x_height_px
        if layout:
            hs = [lb.h for lb in layout if lb.h > 4]
            if hs:
                x_h = float(np.median(hs)) / max(profile.metrics.ascender_ratio, 1.0)
        return InkResult(
            engine="paragraph_ldm", text=text, image=ink, lines=layout,
            x_height=x_h, seed=seed,
            meta={"steps": steps, "guidance": LDM_GUIDANCE,
                  "condLines": len(lines), "quality": quality},
        )


engine = ParagraphLDMEngine()
