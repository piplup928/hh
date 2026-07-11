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
# OFFICIAL release: github.com/awei669/DiffInk (ICLR 2026). Weights + char
# dict are distributed via the authors' Google Drive folder (see their README):
#   https://drive.google.com/drive/folders/1h_uLmn-55WmbSBGh1ES8-rftAbDs8riB
DIFFINK_FOLDER_ID="1h_uLmn-55WmbSBGh1ES8-rftAbDs8riB"
OFFICIAL="$WEIGHTS/diffink/official"
mkdir -p "$OFFICIAL"
if [ ! -f "$OFFICIAL/dit.pt" ]; then
  echo ">> Downloading official DiffInk release (Google Drive folder)..."
  python -m gdown --folder "$DIFFINK_FOLDER_ID" -O "$OFFICIAL/_download" --remaining-ok || {
    echo "!! gdown folder download failed. Download manually from the link above"
    echo "   (or Baidu: https://pan.baidu.com/s/1NhEoO_hIDOn2dC4qN1oe1A?pwd=ddra)"
  }
  # normalize names: pick the newest vae_/dit_ checkpoints + the char dict
  if [ -d "$OFFICIAL/_download" ]; then
    find "$OFFICIAL/_download" -name 'vae*epoch*.pt'  | sort | tail -1 | xargs -I{} cp {} "$OFFICIAL/vae.pt" || true
    find "$OFFICIAL/_download" -name 'dit*epoch*.pt'  | sort | tail -1 | xargs -I{} cp {} "$OFFICIAL/dit.pt" || true
    find "$OFFICIAL/_download" -name 'All_zi.json'    | head -1 | xargs -I{} cp {} "$OFFICIAL/All_zi.json" || true
  fi
  if [ -f "$OFFICIAL/dit.pt" ] && [ -f "$OFFICIAL/vae.pt" ] && [ -f "$OFFICIAL/All_zi.json" ]; then
    echo ">> Official DiffInk ready: $OFFICIAL/{vae.pt,dit.pt,All_zi.json}"
  else
    cat <<EOF
!! Could not assemble $OFFICIAL/{vae.pt,dit.pt,All_zi.json} automatically.
   Place them manually:
     vae.pt      <- the authors' vae_epoch_*.pt   (InkVAE)
     dit.pt      <- the authors' dit_epoch_*.pt   (InkDiT, fine-tuned)
     All_zi.json <- the character dictionary from datas/meta/
EOF
  fi
else
  echo ">> Official DiffInk weights already present."
fi

# Optional: the platform also bundles a scratch InkVAE/InkDiT implementation
# you can train yourself on MathWriting/CROHME for wider math-symbol coverage:
#   python -m app.engine.diffink.train --stage vae --data /path/to/mathwriting
#   python -m app.engine.diffink.train --stage dit --data /path/to/mathwriting

echo ">> Done."
