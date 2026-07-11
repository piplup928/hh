"""Latent DDPM/DDIM machinery for InkDiT (epsilon-prediction, cosine schedule)."""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch


def cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(1e-8, 0.999).float()


class LatentInkDiffusion:
    def __init__(self, T: int = 1000, device: str = "cpu"):
        self.T = T
        self.device = device
        self.betas = cosine_betas(T).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alphas_bar[t].view(-1, 1, 1)
        return ab.sqrt() * z0 + (1 - ab).sqrt() * noise

    def training_loss(self, denoiser: Callable, z0: torch.Tensor, cond_kwargs: dict) -> torch.Tensor:
        B = z0.shape[0]
        t = torch.randint(0, self.T, (B,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        eps = denoiser(z_t, t, **cond_kwargs)
        return torch.nn.functional.mse_loss(eps, noise)

    @torch.no_grad()
    def ddim_sample(self, denoiser: Callable, shape, steps: int = 50,
                    eta: float = 0.0, guidance_scale: float = 1.0,
                    cond_kwargs: Optional[dict] = None,
                    uncond_kwargs: Optional[dict] = None,
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
        device = self.device
        z = torch.randn(shape, device=device, generator=generator)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        for i, t in enumerate(ts):
            t_b = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
            eps = denoiser(z, t_b, **(cond_kwargs or {}))
            if guidance_scale != 1.0 and uncond_kwargs is not None:
                eps_u = denoiser(z, t_b, **uncond_kwargs)
                eps = eps_u + guidance_scale * (eps - eps_u)
            ab_t = self.alphas_bar[t]
            ab_prev = self.alphas_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            z0_pred = (z - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
            dir_zt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            z = ab_prev.sqrt() * z0_pred + dir_zt
            if eta > 0:
                z = z + sigma * torch.randn(shape, device=device, generator=generator)
        return z
