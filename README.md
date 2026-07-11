# ✍ Handwriting Intelligence Platform

An AI-powered text-to-handwriting platform: upload samples of your own
handwriting, let the system learn the complete writing style, then type
normally in a Word-like editor — every character appears as realistic
handwritten ink in **your** style, generated live by diffusion models.

## How it works

```
 upload samples ──► preprocess ──► style analysis ──► StyleProfile
      │              (deskew,        (slant, stroke     (canvases +
      │               binarize,       width, x-height,   metrics +
      │               768×768)        spacing, drift)    reference lines)
      │
 type in editor ──► content router ─┬─ prose ──► Paragraph-LDM (zero-shot
      ▲                             │            paragraph imitation)
      │                             └─ math  ──► DiffInk (InkVAE + InkDiT)
      │                                   │
 ink overlay ◄── style-preserving ◄── ink cache
 (live canvas)    rich-text transforms
                                          │
                  benchmark runner ◄──────┘   HWD · style-mAP · CER
                  (official metrics)          vs. YOUR uploaded samples
```

### Generation models (real, vendored, wired end-to-end)

| Content | Model | Source |
|---|---|---|
| Prose, paragraphs, words | **Paragraph-LDM** — *Zero-Shot Paragraph-level Handwriting Imitation with Latent Diffusion Models* ([arXiv:2409.00786](https://arxiv.org/abs/2409.00786)) | vendored official repo → `backend/vendor/paragraph_handwriting_imitation_ldm` |
| Numbers, equations, fractions, integrals, Greek symbols, chemistry, scientific notation, math punctuation | **DiffInk** — *Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation* ([arXiv:2509.23624](https://arxiv.org/abs/2509.23624)) | faithful implementation (InkVAE + InkDiT) → `backend/app/engine/diffink` (no official code release exists) |

Both models are conditioned zero-shot on the user's uploaded style: the LDM's
style encoder consumes the raw 768×768 style canvas directly; DiffInk is
conditioned on the extracted style vector and its output is rendered with the
user's pen model (stroke width, pressure jitter, slant, ink darkness).

### Real-time pipeline

1. User uploads handwriting → preprocessing → style profile (one-time, seconds).
2. The editor (TipTap/ProseMirror) renders documents natively — caret,
   selection, alignment, justify, tables, lists, image wrap all behave like
   Word/Google Docs. Glyphs are invisible; ink is drawn above.
3. On every edit, text is split into mark-runs, content-addressed, and
   requested over a WebSocket. The cache guarantees retyping/reflowing never
   re-runs diffusion; only changed runs generate.
4. The router sends prose to Paragraph-LDM (page-chunked to its 768×768
   canvas) and mathematical content to DiffInk (unicode → LaTeX normalized).
5. Generated ink returns with per-line/per-word boxes; the overlay draws each
   word aligned to its live DOM rect — so editing is never blocked.
6. Formatting is applied to ink, not fonts, preserving the learned style:
   bold = pressure-scaled dilation, italic = extra shear, color = alpha-channel
   recolor, underline = drift-jittered pen stroke, size = geometric scale.

### Benchmarks — real metrics only

`POST /api/benchmarks/run` generates fresh samples and scores them **against
the user's actual uploaded handwriting**:

* **HWD** (Handwriting Distance) — official [aimagelab/HWD](https://github.com/aimagelab/HWD)
  implementation (Pippi et al., BMVC 2023), vendored at `backend/vendor/HWD`.
* **Style Preservation mAP** — writer-retrieval mean Average Precision on the
  official HWD VGG16 backbone features: generated queries retrieve the user's
  real lines out of a gallery with real IAM distractor writers.
* **FID / BFID / KID / CER** — official HWD-package implementations (optional).

The runner **refuses** to score anything not produced by the real models
(HTTP 409) — no fake numbers, ever.

## Repository layout

```
backend/
  app/
    api/          styles · generate (REST+WS) · documents · benchmarks
    engine/       paragraph_ldm.py · diffink/ · router.py · service.py
                  style/ (preprocess, analyzer, profile) · ink_ops.py · cache.py
    benchmarks/   runner.py · style_map.py
  vendor/         paragraph_handwriting_imitation_ldm/ · HWD/   (official code)
  scripts/        setup_models.sh   (weights download)
  tests/          21 unit tests + e2e smoke
frontend/
  src/editor/     HandwritingEditor · InkOverlay (live ink renderer) · extensions
  src/components/ Toolbar · Style/Paper/Typography/Benchmark panels
docs/ARCHITECTURE.md
```

## Quickstart

### Backend
```bash
cd backend
pip install -r requirements.txt
pip install -e vendor/HWD                # official HWD metric package
bash scripts/setup_models.sh             # downloads Paragraph-LDM checkpoint (GPU: ≥8GB VRAM)
uvicorn app.main:app --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev                              # http://localhost:5173
```

### Engine modes
* `HW_ENGINE_MODE=live` — only real model output; requests fail loudly if
  weights are missing.
* `HW_ENGINE_MODE=preview` (default for development) — when weights are absent
  the editor stays usable via a clearly-labelled font preview. Preview output
  is tagged `engine: "preview"` everywhere and is rejected by benchmarks.

### Model weights
* **Paragraph-LDM**: pre-trained checkpoint from the authors
  ([Google Drive link in their README](https://github.com/M4rt1nM4yr/paragraph_handwriting_imitation_ldm))
  → `backend/weights/paragraph_ldm/ldm.ckpt` (setup script does this).
* **DiffInk**: no public checkpoint exists; train the bundled implementation on
  [MathWriting](https://arxiv.org/abs/2404.10690) or CROHME:
  ```bash
  python -m app.engine.diffink.train --stage vae --data /path/to/mathwriting
  python -m app.engine.diffink.train --stage dit --data /path/to/mathwriting
  ```

### Docker
```bash
docker compose up --build     # backend :8000 (GPU passthrough), frontend :5173
```

## Editor features

Rich text: bold, italic, underline, headings, paragraphs, text color,
highlight color, font size, line height, letter/word/paragraph spacing,
align left/center/right, justify, bullet/numbered/multi-level lists.
Tables: insert, custom rows/columns, merge/split cells, column resize,
borders, padding — handwriting is placed inside each cell respecting
alignment. Images: upload, resize, rotate, text wrap (left/right/inline) —
ink flows around wrapped images. Paper simulation: Plain, College/Wide/
Narrow/Fine Ruled, 5mm Squared, Graph, Dot Grid, Blank Notebook, Custom —
with width, height, margins, page color, line color, line thickness and
grid size controls. Print/PDF export via the print stylesheet.

## Tests

```bash
cd backend && python -m pytest tests/   # 21 tests
```
