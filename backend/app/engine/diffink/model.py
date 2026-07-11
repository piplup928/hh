"""DiffInk neural modules: InkVAE + InkDiT (arXiv:2509.23624).

Trajectory representation: sequences of (dx, dy, pen) where pen ∈ {0,1}
(1 = pen down / drawing). Coordinates are normalized offsets.

InkVAE: Transformer encoder pools the trajectory into a temporally
down-sampled latent token sequence (stride S), KL-regularized; a Transformer
decoder reconstructs the full trajectory (MSE on offsets + BCE on pen state).

InkDiT: DiT-style latent denoiser. Each block =
  self-attention -> cross-attention(glyph tokens) -> MLP,
all modulated by AdaLN-Zero on (diffusion timestep ⊕ style vector).
Style vector = learned projection of the platform's 12-dim StyleMetrics
(slant, stroke width, x-height, spacing stats …) so generated math matches
the handwriting style learned from the user's uploads.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PAD_ID, vocab_size

STYLE_DIM = 12  # interpretable style metrics vector (see style/analyzer.py)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ================================================================== InkVAE ===
class InkVAE(nn.Module):
    def __init__(self, d_model: int = 256, latent_dim: int = 64, stride: int = 4,
                 n_layers: int = 4, n_heads: int = 8, max_len: int = 2048):
        super().__init__()
        self.stride = stride
        self.latent_dim = latent_dim
        self.in_proj = nn.Linear(3, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.1, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.to_mu = nn.Linear(d_model * stride, latent_dim)
        self.to_logvar = nn.Linear(d_model * stride, latent_dim)

        self.lat_proj = nn.Linear(latent_dim, d_model * stride)
        dec_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.1, batch_first=True,
            norm_first=True, activation="gelu")
        self.decoder = nn.TransformerEncoder(dec_layer, n_layers)
        self.out_head = nn.Linear(d_model, 3)  # dx, dy, pen-logit

    # T must be a multiple of stride (pad upstream)
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = self.encoder(self.pos(self.in_proj(x)), src_key_padding_mask=mask)
        B, T, D = h.shape
        h = h.reshape(B, T // self.stride, D * self.stride)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B, L, _ = z.shape
        h = self.lat_proj(z).reshape(B, L * self.stride, -1)
        h = self.decoder(self.pos(h))
        return self.out_head(h)

    def forward(self, x, mask=None):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss(recon, target, mu, logvar, kl_weight: float = 1e-3):
        offs_loss = F.mse_loss(recon[..., :2], target[..., :2])
        pen_loss = F.binary_cross_entropy_with_logits(recon[..., 2], target[..., 2])
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return offs_loss + 0.5 * pen_loss + kl_weight * kl, {
            "offset": offs_loss.item(), "pen": pen_loss.item(), "kl": kl.item()}


# ================================================================== InkDiT ===
class AdaLNZero(nn.Module):
    """AdaLN-Zero modulation producing shift/scale/gate triplets."""

    def __init__(self, d_model: int, n_chunks: int = 9):
        super().__init__()
        self.n_chunks = n_chunks
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * n_chunks))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, cond: torch.Tensor):
        return self.mod(cond).chunk(self.n_chunks, dim=-1)


class InkDiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.ada = AdaLNZero(d_model, 9)

    def forward(self, x, glyphs, cond, glyph_pad_mask=None):
        (s1, b1, g1, s2, b2, g2, s3, b3, g3) = self.ada(cond)
        s1, b1, g1 = s1.unsqueeze(1), b1.unsqueeze(1), g1.unsqueeze(1)
        s2, b2, g2 = s2.unsqueeze(1), b2.unsqueeze(1), g2.unsqueeze(1)
        s3, b3, g3 = s3.unsqueeze(1), b3.unsqueeze(1), g3.unsqueeze(1)

        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + s2) + b2
        x = x + g2 * self.cross(h, glyphs, glyphs, need_weights=False,
                                key_padding_mask=glyph_pad_mask)[0]
        h = self.norm3(x) * (1 + s3) + b3
        x = x + g3 * self.mlp(h)
        return x


class InkDiT(nn.Module):
    def __init__(self, latent_dim: int = 64, d_model: int = 384, n_layers: int = 8,
                 n_heads: int = 6, max_latent_len: int = 512,
                 n_writers: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_latent_len)
        self.glyph_emb = nn.Embedding(vocab_size(), d_model, padding_idx=PAD_ID)
        self.glyph_pos = PositionalEncoding(d_model, 512)
        self.t_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(),
                                   nn.Linear(d_model, d_model))
        # style conditioning: interpretable metrics vector (inference & training)
        # + optional writer-id embedding (training regularizer, id 0 = unknown)
        self.style_proj = nn.Sequential(nn.Linear(STYLE_DIM, d_model), nn.SiLU(),
                                        nn.Linear(d_model, d_model))
        self.writer_emb = nn.Embedding(n_writers, d_model)
        nn.init.zeros_(self.writer_emb.weight)
        self.blocks = nn.ModuleList(InkDiTBlock(d_model, n_heads) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_ada = AdaLNZero(d_model, 2)
        self.out = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.d_model = d_model

    def forward(self, z_t, t, glyph_ids, style_vec, writer_ids=None):
        """z_t: (B,L,latent) noisy latents; t: (B,) timesteps;
        glyph_ids: (B,G) LaTeX token ids; style_vec: (B,STYLE_DIM)."""
        cond = self.t_mlp(timestep_embedding(t, self.d_model))
        cond = cond + self.style_proj(style_vec)
        if writer_ids is not None:
            cond = cond + self.writer_emb(writer_ids)
        glyphs = self.glyph_pos(self.glyph_emb(glyph_ids))
        pad_mask = glyph_ids.eq(PAD_ID)
        x = self.pos(self.in_proj(z_t))
        for blk in self.blocks:
            x = blk(x, glyphs, cond, glyph_pad_mask=pad_mask)
        shift, scale = self.final_ada(cond)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.out(x)
