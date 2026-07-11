"""DiffInk training — two stages, mirroring the paper:

  stage vae : train InkVAE (trajectory reconstruction + KL)
  stage dit : freeze InkVAE, train InkDiT on latent epsilon-prediction with
              glyph cross-attention + style AdaLN conditioning and 10%
              glyph-dropout for classifier-free guidance.

Usage:
  python -m app.engine.diffink.train --stage vae --data /path/to/mathwriting --epochs 30
  python -m app.engine.diffink.train --stage dit --data /path/to/mathwriting --epochs 60

Checkpoints land in backend/weights/diffink/ (inkvae.pt / inkdit.pt) where the
inference engine picks them up automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ...config import DIFFINK_DIT_CKPT, DIFFINK_VAE_CKPT, resolve_device
from .data import InkMLDataset, collate
from .diffusion import LatentInkDiffusion
from .model import InkDiT, InkVAE
from .tokenizer import PAD_ID


def train_vae(args, device):
    ds = InkMLDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True)
    hparams = dict(d_model=256, latent_dim=64, stride=4, n_layers=4)
    vae = InkVAE(**hparams).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        for step, (traj, _g, _s, _w) in enumerate(dl):
            traj = traj.to(device)
            recon, mu, logvar = vae(traj)
            loss, parts = InkVAE.loss(recon, traj, mu, logvar)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                print(f"[vae] epoch {epoch} step {step} loss {loss.item():.4f} {parts}")
        DIFFINK_VAE_CKPT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": vae.state_dict(), "hparams": hparams},
                   DIFFINK_VAE_CKPT)
        print(f"[vae] saved -> {DIFFINK_VAE_CKPT}")


def train_dit(args, device):
    vae_sd = torch.load(DIFFINK_VAE_CKPT, map_location="cpu")
    vae = InkVAE(**vae_sd.get("hparams", {}))
    vae.load_state_dict(vae_sd["state_dict"])
    vae.to(device).eval().requires_grad_(False)

    ds = InkMLDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True)
    hparams = dict(latent_dim=vae.latent_dim, d_model=384, n_layers=8, n_heads=6)
    dit = InkDiT(**hparams).to(device)
    diff = LatentInkDiffusion(device=device)
    opt = torch.optim.AdamW(dit.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        for step, (traj, glyphs, style, writers) in enumerate(dl):
            traj, glyphs = traj.to(device), glyphs.to(device)
            style, writers = style.to(device), writers.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(traj)
                z0 = vae.reparameterize(mu, logvar)
            # classifier-free guidance: drop glyph conditioning on 10% of rows
            drop = torch.rand(glyphs.shape[0], device=device) < 0.1
            glyphs = glyphs.masked_fill(drop.unsqueeze(1), PAD_ID)
            loss = diff.training_loss(
                lambda z_t, t, **kw: dit(z_t, t, **kw), z0,
                {"glyph_ids": glyphs, "style_vec": style, "writer_ids": writers})
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                print(f"[dit] epoch {epoch} step {step} loss {loss.item():.5f}")
        DIFFINK_DIT_CKPT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": dit.state_dict(), "hparams": hparams},
                   DIFFINK_DIT_CKPT)
        print(f"[dit] saved -> {DIFFINK_DIT_CKPT}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["vae", "dit"], required=True)
    p.add_argument("--data", required=True, help="root dir containing .inkml files")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    device = resolve_device()
    print(f"training stage={args.stage} on {device}")
    (train_vae if args.stage == "vae" else train_dit)(args, device)


if __name__ == "__main__":
    main()
