"""DiffInk — glyph- and style-aware latent diffusion for online handwriting.

Implementation of the DiffInk architecture (arXiv:2509.23624):
  * InkVAE — sequential KL-VAE compressing pen trajectories (dx, dy, pen)
    into a shorter latent token sequence.
  * InkDiT — conditional latent diffusion Transformer denoising latent ink
    tokens, conditioned on glyph tokens (LaTeX for math) via cross-attention
    and on a writer-style vector via AdaLN.

The paper has no official public code/checkpoint release, so this package is
a faithful re-implementation intended to be trained on online math corpora
(Google MathWriting, CROHME). See train.py.
"""
from .engine import DiffInkEngine, engine  # noqa: F401
