"""DiffInk inference engine implementing the HandwritingEngine contract."""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from ...config import DIFFINK_DIT_CKPT, DIFFINK_VAE_CKPT, resolve_device
from ..base import EngineUnavailable, InkResult, LineBox
from ..style.profile import StyleStore, store as style_store
from . import render
from .diffusion import LatentInkDiffusion
from .model import STYLE_DIM, InkDiT, InkVAE
from .tokenizer import PAD_ID, encode

log = logging.getLogger("engine.diffink")


def style_metrics_to_vector(metrics) -> "np.ndarray":
    """StyleMetrics -> normalized 12-dim conditioning vector (order is part of
    the trained-model contract; keep in sync with train.py)."""
    m = metrics
    return np.array([
        m.slant_deg / 45.0,
        m.stroke_width_px / 10.0,
        m.stroke_width_std / 5.0,
        m.x_height_px / 100.0,
        m.ascender_ratio / 3.0,
        m.line_spacing_px / 200.0,
        m.word_spacing_px / 80.0,
        m.intra_word_gap_px / 30.0,
        m.baseline_drift_px / 10.0,
        m.baseline_slope_deg / 10.0,
        m.ink_darkness,
        m.density * 10.0,
    ], dtype=np.float32)


class DiffInkEngine:
    name = "diffink"

    def __init__(self, vae_ckpt=DIFFINK_VAE_CKPT, dit_ckpt=DIFFINK_DIT_CKPT,
                 styles: StyleStore = style_store):
        self.vae_ckpt = vae_ckpt
        self.dit_ckpt = dit_ckpt
        self.styles = styles
        self.device = resolve_device()
        self._vae: Optional[InkVAE] = None
        self._dit: Optional[InkDiT] = None
        self._diffusion: Optional[LatentInkDiffusion] = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    def is_ready(self) -> bool:
        if self._dit is not None:
            return True
        return self.vae_ckpt.exists() and self.dit_ckpt.exists() and self._load_error is None

    def status(self) -> dict:
        return {
            "engine": self.name,
            "loaded": self._dit is not None,
            "vaeCheckpoint": str(self.vae_ckpt),
            "ditCheckpoint": str(self.dit_ckpt),
            "checkpointPresent": self.vae_ckpt.exists() and self.dit_ckpt.exists(),
            "device": self.device,
            "error": self._load_error,
        }

    def _ensure_model(self):
        if self._dit is not None:
            return
        with self._lock:
            if self._dit is not None:
                return
            if not (self.vae_ckpt.exists() and self.dit_ckpt.exists()):
                raise EngineUnavailable(
                    "DiffInk weights missing (weights/diffink/inkvae.pt, inkdit.pt). "
                    "Train them with `python -m app.engine.diffink.train` "
                    "on MathWriting/CROHME — see backend/scripts/setup_models.sh."
                )
            try:
                import torch

                vae_sd = torch.load(str(self.vae_ckpt), map_location="cpu")
                dit_sd = torch.load(str(self.dit_ckpt), map_location="cpu")
                vae = InkVAE(**vae_sd.get("hparams", {}))
                vae.load_state_dict(vae_sd["state_dict"])
                dit = InkDiT(**dit_sd.get("hparams", {}))
                dit.load_state_dict(dit_sd["state_dict"])
                vae.to(self.device).eval()
                dit.to(self.device).eval()
                self._vae, self._dit = vae, dit
                self._diffusion = LatentInkDiffusion(device=self.device)
                log.info("DiffInk loaded on %s", self.device)
            except Exception as e:  # noqa: BLE001
                self._load_error = f"{type(e).__name__}: {e}"
                log.exception("DiffInk load failed")
                raise EngineUnavailable(self._load_error) from e

    def generate(self, latex: str, style_id: str, *, seed: Optional[int] = None,
                 quality: str = "live") -> InkResult:
        import torch

        profile = self.styles.get(style_id)
        if profile is None:
            raise KeyError(f"unknown style {style_id}")
        self._ensure_model()

        glyph_ids = torch.tensor([encode(latex)], dtype=torch.long, device=self.device)
        style_vec = torch.from_numpy(
            style_metrics_to_vector(profile.metrics)).unsqueeze(0).to(self.device)

        # latent length heuristic: ~12 latent tokens per glyph token
        n_glyphs = int((glyph_ids != PAD_ID).sum().item())
        L = int(np.clip(n_glyphs * 12, 32, 480))
        steps = 30 if quality == "live" else 80
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)

        with self._lock, torch.no_grad():
            dit, vae, diff = self._dit, self._vae, self._diffusion
            uncond_ids = torch.full_like(glyph_ids, PAD_ID)
            z = diff.ddim_sample(
                lambda z_t, t, **kw: dit(z_t, t, **kw),
                shape=(1, L, dit.latent_dim), steps=steps, guidance_scale=2.0,
                cond_kwargs={"glyph_ids": glyph_ids, "style_vec": style_vec},
                uncond_kwargs={"glyph_ids": uncond_ids, "style_vec": style_vec},
                generator=gen,
            )
            traj = vae.decode(z)[0].cpu().numpy()  # (T,3) dx,dy,pen-logit
        traj[:, 2] = 1.0 / (1.0 + np.exp(-traj[:, 2]))  # sigmoid pen

        m = profile.metrics
        strokes = render.offsets_to_strokes(traj)
        target_h = m.x_height_px * m.ascender_ratio
        strokes = render.normalize_strokes(strokes, target_h)
        # residual shear: bring generated slant fully onto the user's slant
        strokes = render.shear_strokes(strokes, m.slant_deg * 0.35, target_h)
        svg, _ = render.strokes_to_svg(strokes, m.stroke_width_px)
        img = render.rasterize_strokes(
            strokes, m.stroke_width_px, width_jitter=m.stroke_width_std,
            darkness=m.ink_darkness)

        return InkResult(
            engine="diffink", text=latex, image=img, svg=svg,
            x_height=m.x_height_px, seed=seed,
            lines=[LineBox(text=latex, x=0, y=0, w=img.width, h=img.height,
                           baseline=int(img.height * 0.8))],
            meta={"steps": steps, "latentLen": L, "quality": quality},
        )


engine = DiffInkEngine()
