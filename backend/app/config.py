"""Central configuration for the handwriting platform backend."""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = BACKEND_ROOT / "vendor"
PARAGRAPH_LDM_DIR = VENDOR_DIR / "paragraph_handwriting_imitation_ldm"
HWD_DIR = VENDOR_DIR / "HWD"

WEIGHTS_DIR = Path(os.environ.get("HW_WEIGHTS_DIR", BACKEND_ROOT / "weights"))
PARAGRAPH_LDM_CKPT = Path(
    os.environ.get("PARAGRAPH_LDM_CKPT", WEIGHTS_DIR / "paragraph_ldm" / "ldm.ckpt")
)
DIFFINK_VAE_CKPT = Path(os.environ.get("DIFFINK_VAE_CKPT", WEIGHTS_DIR / "diffink" / "inkvae.pt"))
DIFFINK_DIT_CKPT = Path(os.environ.get("DIFFINK_DIT_CKPT", WEIGHTS_DIR / "diffink" / "inkdit.pt"))

DATA_DIR = Path(os.environ.get("HW_DATA_DIR", BACKEND_ROOT / "data"))
STYLES_DIR = DATA_DIR / "styles"
DOCS_DIR = DATA_DIR / "documents"
CACHE_DIR = DATA_DIR / "cache"
BENCH_DIR = DATA_DIR / "benchmarks"

for _d in (STYLES_DIR, DOCS_DIR, CACHE_DIR, BENCH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Engine mode:
#   live    -> only real model output; requests fail loudly when weights are missing
#   preview -> when a model is unavailable, a clearly-labelled vector-font preview is
#              returned so the editor stays usable during development. Preview output
#              is tagged engine="preview" end-to-end and is REJECTED by the benchmark
#              runner — benchmarks only ever score real model output.
ENGINE_MODE = os.environ.get("HW_ENGINE_MODE", "preview").lower()

DEVICE = os.environ.get("HW_DEVICE", "")  # "" = auto (cuda if available)

# Paragraph-LDM canvas (fixed by the pre-trained model)
LDM_CANVAS = 768
# Empirically the released 768x768 model handles ~8-10 lines / ~42 chars per line.
LDM_MAX_CHARS_PER_LINE = int(os.environ.get("HW_LDM_CHARS_PER_LINE", 42))
LDM_MAX_LINES = int(os.environ.get("HW_LDM_MAX_LINES", 8))
LDM_STEPS = int(os.environ.get("HW_LDM_STEPS", 50))          # DDIM steps (live typing)
LDM_STEPS_FINAL = int(os.environ.get("HW_LDM_STEPS_FINAL", 150))  # export quality
LDM_GUIDANCE = float(os.environ.get("HW_LDM_GUIDANCE", 2.5))

# Generation cache
CACHE_MAX_ITEMS = int(os.environ.get("HW_CACHE_MAX_ITEMS", 4096))

def resolve_device() -> str:
    if DEVICE:
        return DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
