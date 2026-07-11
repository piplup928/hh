"""Adapter for the OFFICIAL DiffInk release (github.com/awei669/DiffInk).

Loads the authors' pretrained InkVAE + InkDiT checkpoints (ICLR 2026,
trained on CASIA-OLHWDB / IAM-OnDB) vendored at backend/vendor/DiffInk and
generates pen trajectories for arbitrary text.

Inference modes
---------------
The official `val_dit.py` is a prefix-infilling evaluator: its "style
reference" is the VAE-encoded prefix of the very trajectory being
regenerated. This adapter generalizes that mechanism:

* text-only  — cond mask fully open (everything generated); conditioning is
  the text via cross-modal embedding (their CFG cond/uncond branches then
  coincide, which is exactly their `drop_cond` semantics). Used when the user
  only uploaded offline images: glyph formation comes from the model, writer
  identity is applied by the platform's pen model (stroke width, slant,
  darkness from the analyzed StyleProfile).
* prefix-ref — when an ONLINE reference trajectory exists for the style
  (e.g. recorded on a tablet, stored as `online_reference.json` in the style
  dir), it is VAE-encoded and locked as the prefix (`latent_mask=0` there),
  reproducing the paper's true style-transfer conditioning.

Weight layout (produced by scripts/setup_models.sh from the authors' release):
    weights/diffink/official/vae.pt        (their vae_epoch_*.pt)
    weights/diffink/official/dit.pt        (their dit_epoch_*.pt, fine-tuned)
    weights/diffink/official/All_zi.json   (char dict — key ORDER defines ids)
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ...config import VENDOR_DIR, WEIGHTS_DIR, resolve_device

log = logging.getLogger("engine.diffink.official")

DIFFINK_DIR = VENDOR_DIR / "DiffInk"
OFFICIAL_DIR = Path(WEIGHTS_DIR) / "diffink" / "official"
VAE_CKPT = OFFICIAL_DIR / "vae.pt"
DIT_CKPT = OFFICIAL_DIR / "dit.pt"
CHAR_DICT = OFFICIAL_DIR / "All_zi.json"

POINTS_PER_CHAR = 48        # sequence-length heuristic (chars -> ink points)
COMPRESSION = 8             # VAE temporal downsampling (3x stride-2)


def _import_vendor():
    root = str(DIFFINK_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


class OfficialDiffInk:
    """Thin, device-agnostic wrapper around the official models."""

    def __init__(self):
        self.device = resolve_device()
        self._vae = None
        self._dit = None
        self._diffusion = None
        self._chars: Optional[dict] = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ state
    def weights_present(self) -> bool:
        return VAE_CKPT.exists() and DIT_CKPT.exists() and CHAR_DICT.exists()

    def is_ready(self) -> bool:
        if self._dit is not None:
            return True
        return self.weights_present() and self._load_error is None

    def status(self) -> dict:
        return {
            "backend": "official (awei669/DiffInk, ICLR 2026)",
            "loaded": self._dit is not None,
            "weightsDir": str(OFFICIAL_DIR),
            "checkpointPresent": self.weights_present(),
            "charDictSize": len(self._chars) if self._chars else None,
            "device": self.device,
            "error": self._load_error,
        }

    # ------------------------------------------------------------------- load
    def _ensure_model(self):
        if self._dit is not None:
            return
        with self._lock:
            if self._dit is not None:
                return
            if not self.weights_present():
                raise RuntimeError(
                    f"official DiffInk weights missing under {OFFICIAL_DIR} "
                    "(vae.pt, dit.pt, All_zi.json). Run scripts/setup_models.sh"
                )
            try:
                import torch

                _import_vendor()
                from model.vae import VAE                       # vendor
                from model.dit import DiT                       # vendor
                from model.diffusion import Diffusion           # vendor
                from utils.utils import ModelConfig, load_config_from_yaml  # vendor

                cfg = load_config_from_yaml(str(DIFFINK_DIR / "configs" / "dit_val_config.yaml"))
                chars = json.loads(CHAR_DICT.read_text(encoding="utf-8"))
                # key ORDER defines the trained embedding ids (see their
                # dataset.read_all_chars); +1 because id 0 is the pad/filler
                self._chars = {k: i for i, k in enumerate(chars.keys())}
                cfg["num_text_embedding"] = len(self._chars) + 1
                config = ModelConfig(cfg)

                def _load(path):
                    ckpt = torch.load(str(path), map_location="cpu")
                    sd = ckpt.get("model_state_dict", ckpt)
                    if any(k.startswith("module.") for k in sd):
                        sd = {k[len("module."):]: v for k, v in sd.items()}
                    return sd

                vae = VAE(config)
                vae_sd = _load(VAE_CKPT)
                model_sd = vae.state_dict()
                filtered = {k: v for k, v in vae_sd.items()
                            if k in model_sd and v.shape == model_sd[k].shape}
                missing, _ = vae.load_state_dict(filtered, strict=False)
                core_missing = [k for k in missing
                                if not k.startswith(("ocr_model", "style_classifier"))]
                if core_missing:
                    raise RuntimeError(
                        f"VAE checkpoint mismatch, core keys missing: {core_missing[:4]}")

                dit = DiT(config)
                dit.load_state_dict(_load(DIT_CKPT))

                vae.to(self.device).eval()
                dit.to(self.device).eval()
                self._vae, self._dit = vae, dit
                self._diffusion = Diffusion(
                    noise_steps=1000, schedule_type="cosine", device=self.device)
                log.info("official DiffInk loaded on %s (%d chars)",
                         self.device, len(self._chars))
            except Exception as e:  # noqa: BLE001
                self._load_error = f"{type(e).__name__}: {e}"
                log.exception("official DiffInk load failed")
                raise

    # ------------------------------------------------------------------- text
    def encode_text(self, text: str) -> Tuple["object", List[str]]:
        """Text -> [1, n] long tensor of trained char ids.

        Unknown characters are dropped (and reported) rather than crashing —
        the released dict is CASIA-based; extend coverage by fine-tuning with
        their tune_dit script on a math corpus."""
        import torch

        assert self._chars is not None
        known, skipped = [], []
        for ch in text:
            if ch in self._chars:
                known.append(self._chars[ch])
            elif ch == " " and "," in self._chars:
                continue  # spacing is implicit in trajectory
            else:
                skipped.append(ch)
        if not known:
            raise ValueError(
                f"no characters of {text!r} are covered by the released "
                f"DiffInk char dict ({len(skipped)} unsupported); fine-tune "
                "with vendor/DiffInk tune scripts for this alphabet")
        # ValDataset appends the ideographic comma as terminator when present
        if "、" in self._chars:
            known.append(self._chars["、"])
        return torch.tensor([known], dtype=torch.long, device=self.device), skipped

    # -------------------------------------------------------------- reference
    def _encode_reference(self, ref_points: np.ndarray):
        """[T,5] reference trajectory -> latent [1, T/8, latent_dim]."""
        import torch

        T = (len(ref_points) // COMPRESSION) * COMPRESSION
        if T < COMPRESSION:
            return None
        pts = torch.from_numpy(ref_points[:T].astype(np.float32))
        pts = pts.unsqueeze(0).permute(0, 2, 1).to(self.device)  # [1,5,T]
        with torch.no_grad():
            z, _mu, _logvar = self._vae.encode(pts)
        return z.permute(0, 2, 1)  # [1, T/8, latent]

    # --------------------------------------------------------------- generate
    def generate_points(self, text: str, *, steps: int = 20, cfg_scale: float = 1.0,
                        seed: Optional[int] = None,
                        reference_points: Optional[np.ndarray] = None,
                        ) -> Tuple[np.ndarray, dict]:
        """Generate a [T,5] point sequence (dx, dy, pen one-hot x3) for text."""
        import torch

        self._ensure_model()
        text_idx, skipped = self.encode_text(text)
        n_chars = text_idx.shape[1]

        T = int(np.clip(n_chars * POINTS_PER_CHAR, 96, 3200))
        T = (T // COMPRESSION) * COMPRESSION
        L = T // COMPRESSION
        latent_dim = self._vae.conv_mu.out_channels if hasattr(self._vae, "conv_mu") else 384

        with self._lock, torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)

            ref_latent = None
            if reference_points is not None and len(reference_points) >= COMPRESSION:
                ref_latent = self._encode_reference(reference_points)
            if ref_latent is not None:
                Lr = ref_latent.shape[1]
                cond = torch.cat(
                    [ref_latent,
                     torch.zeros(1, L, ref_latent.shape[2], device=self.device)], dim=1)
                latent_mask = torch.ones(1, Lr + L, device=self.device)
                latent_mask[:, :Lr] = 0.0          # lock the reference prefix
                padding_mask = torch.ones(1, Lr + L, device=self.device)
                total_T = (Lr + L) * COMPRESSION
            else:
                Lr = 0
                cond = torch.zeros(1, L, latent_dim, device=self.device)
                latent_mask = torch.ones(1, L, device=self.device)
                padding_mask = torch.ones(1, L, device=self.device)
                total_T = T

            x_pred = self._diffusion.ddim_sample(
                self._dit, 1, cond, text_idx, latent_mask, padding_mask,
                sampling_timesteps=steps, eta=0.0, cfg_scale=cfg_scale)

            # keep locked prefix latents exactly (paper's x_mix remix)
            if Lr:
                m = latent_mask.unsqueeze(-1)
                x_pred = x_pred * m + cond * (1 - m)

            out = self._vae.decode(x_pred.permute(0, 2, 1))  # [1,123,T]

            _import_vendor()
            from model.gmm import get_mixture_coef, sample_from_params  # vendor

            coef = get_mixture_coef(out, num_mixture=20)
            params = [t[0] for t in coef[:7]]
            points = sample_from_params(params, temp=0.1,
                                        max_seq_len=total_T, greedy=True)

        if Lr:  # drop the reference prefix from the rendered output
            points = points[Lr * COMPRESSION:]
        # truncate at the first explicit end-state
        ends = np.nonzero(points[:, 4] > 0.5)[0]
        if ends.size:
            points = points[: max(int(ends[0]), 8)]
        meta = {"backend": "official", "steps": steps, "chars": int(n_chars),
                "skippedChars": skipped, "refPrefix": bool(Lr), "points": int(len(points))}
        return points.astype(np.float32), meta

    @staticmethod
    def points_to_offsets(points: np.ndarray) -> np.ndarray:
        """Official [T,5] (dx,dy,pen-onehot) -> renderer's (dx,dy,pen) format.
        pen column 2 (`is_next`) == 1 means the segment to the next point is
        drawn; our renderer treats pen>=0.5 as pen-down at the point."""
        out = np.zeros((len(points), 3), np.float32)
        out[:, :2] = points[:, :2]
        out[:, 2] = points[:, 2]
        return out


official = OfficialDiffInk()
