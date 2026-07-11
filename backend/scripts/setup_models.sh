#!/usr/bin/env bash
# Downloads / prepares all model weights needed for LIVE generation.
#
#   backend/weights/
#   ├── paragraph_ldm/ldm.ckpt        Paragraph-LDM pre-trained checkpoint (official release)
#   ├── hwd/VGG16_class_10400.pth     HWD backbone (auto-downloaded on first metric run too)
#   └── diffink/inkvae.pt / inkdit.pt DiffInk weights (train with app/engine/diffink/train.py
#                                     on MathWriting/CROHME, or drop in your own)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="$ROOT/weights"
mkdir -p "$WEIGHTS/paragraph_ldm" "$WEIGHTS/hwd" "$WEIGHTS/diffink"

# ---------------------------------------------------------------- Paragraph-LDM
# Official pre-trained checkpoint from the paper authors (README of
# github.com/M4rt1nM4yr/paragraph_handwriting_imitation_ldm):
GDRIVE_ID="1Wu2hh69GN0ib4sZiXkNIIfZRUdZTNSyx"
if [ ! -f "$WEIGHTS/paragraph_ldm/ldm.ckpt" ]; then
  echo ">> Downloading Paragraph-LDM pre-trained checkpoint (~GBs, Google Drive)..."
  python -m gdown "$GDRIVE_ID" -O "$WEIGHTS/paragraph_ldm/ldm.ckpt" || {
    echo "!! gdown failed (Drive quota / network). Download manually:"
    echo "   https://drive.google.com/file/d/$GDRIVE_ID/view"
    echo "   and place it at $WEIGHTS/paragraph_ldm/ldm.ckpt"
  }
else
  echo ">> Paragraph-LDM checkpoint already present."
fi

# ------------------------------------------------------------------------- HWD
HWD_URL="https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth"
if [ ! -f "$WEIGHTS/hwd/VGG16_class_10400.pth" ]; then
  echo ">> Downloading HWD VGG16 backbone..."
  curl -L --fail -o "$WEIGHTS/hwd/VGG16_class_10400.pth" "$HWD_URL" || \
    echo "!! HWD backbone download failed; the hwd package will fetch it lazily at first use."
fi

# --------------------------------------------------------------------- DiffInk
if [ ! -f "$WEIGHTS/diffink/inkdit.pt" ]; then
  cat <<'EOF'
>> DiffInk weights not found (expected: weights/diffink/inkvae.pt + inkdit.pt).
   DiffInk (arXiv:2509.23624) has no public official checkpoint release.
   Train the bundled faithful implementation on MathWriting / CROHME:
     python -m app.engine.diffink.train --stage vae --data /path/to/mathwriting
     python -m app.engine.diffink.train --stage dit --data /path/to/mathwriting
   Until weights exist the math pipeline reports engine "unavailable" (it will
   NEVER silently substitute fake output in benchmark runs).
EOF
fi

echo ">> Done."
