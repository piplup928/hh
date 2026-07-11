# Architecture

## Design principles

1. **The models own the style.** Paragraph-LDM imitates zero-shot from the raw
   768×768 style canvas (its trained interface); DiffInk is conditioned on the
   extracted style vector. Nothing in the platform "post-styles" model output
   beyond the user-requested formatting transforms.
2. **The browser owns layout.** The ProseMirror document renders invisibly and
   natively; the ink overlay only *skins* words at their measured DOM rects.
   That is why every Word-class feature (justify, tables, wrap, lists,
   alignment) works without teaching the models anything about layout.
3. **Diffusion is slow; caching is the product.** Every generated run is
   content-addressed by `(engine, style, text, seed, quality, marks)`. Editing
   only regenerates the runs whose content actually changed.
4. **Benchmarks never lie.** Only official metric implementations, only real
   model output, only the user's real uploads as reference.

## Backend components

| Module | Responsibility |
|---|---|
| `engine/style/preprocess.py` | upload → grayscale → illumination flattening → adaptive binarization → deskew → 768×768 style canvas + line/word segmentation |
| `engine/style/analyzer.py` | slant (shear-search), stroke width (distance transform), x-height, baseline drift/slope (robust lower-envelope regression), word/letter spacing, ink darkness |
| `engine/style/profile.py` | persisted StyleProfile: canvases (LDM conditioning), metrics (DiffInk conditioning + layout), reference line crops (benchmark ground truth) |
| `engine/paragraph_ldm.py` | device-agnostic wrapper over the vendored official sampler: text+style conditioning tensors, DDIM with classifier-free guidance, page→ink alpha extraction, line/word box layout |
| `engine/text_layout.py` | greedy wrap + paragraph chunking to the model's 768×768 / ~8-line capacity |
| `engine/diffink/` | InkVAE (stride-4 sequential KL-VAE over (dx,dy,pen)) + InkDiT (AdaLN-Zero DiT, glyph cross-attention, style conditioning) + cosine-schedule latent DDIM + LaTeX tokenizer + InkML data pipeline + two-stage trainer + pen-model renderer (SVG + raster) |
| `engine/router.py` | text/math segmentation: `$...$`, numerics, scientific notation, chemistry formulas, unicode math symbols → DiffInk; everything else → Paragraph-LDM; unicode→LaTeX normalization |
| `engine/ink_ops.py` | style-preserving formatting: dilation bold, shear italic, alpha recolor, jittered underline, geometric scaling |
| `engine/cache.py` | memory LRU + disk PNG/JSON persistence |
| `engine/service.py` | orchestration, thread-pool executors around GPU sampling, engine selection with explicit live/preview policy |
| `benchmarks/runner.py` | builds FolderDataset layouts (real refs + IAM distractor writers), runs HWD/FID/BFID/KID/CER + style-mAP, integrity guard |
| `benchmarks/style_map.py` | writer-retrieval mAP on HWD backbone features |

## Real-time protocol

```
WS /api/generate/ws
  → {"op":"gen","reqId":7,"text":"...","styleId":"...","marks":{"bold":true},"quality":"live"}
  ← {"op":"ink","reqId":7,"results":[{engine,png,svg,lines[{words[...]}],xHeight,...}]}
```

Quality ladder: `live` = 50 DDIM steps (typing), `final` = 150 steps
(export/benchmarks). The frontend debounces edits ~120ms, drops stale
responses by request id, and draws placeholder glyphs until ink arrives.

## Frontend ink overlay

`InkOverlay.tsx` walks the ProseMirror doc → mark-runs → tokens with document
positions. For each run it requests ink once (content-addressed). Draw pass:
per-token `coordsAtPos` rects; if the model's word boxes match the token
count, each word sprite is placed word-precisely (scaled by rect height,
aspect preserved, width-capped); otherwise ink lines are stretched across the
DOM line groups. Underlines are drawn as jittered pen strokes on canvas.

## Failure modes & honesty

* Missing Paragraph-LDM checkpoint: `live` mode → HTTP 503 with setup
  instructions; `preview` mode → labelled font preview (never benchmarked).
* Missing DiffInk weights: same policy; training scripts are bundled since no
  official checkpoint release exists for the paper.
* Word segmentation mismatch between model output and transcript: layout
  degrades gracefully to proportional line-strip placement — style is intact,
  only word-box precision is reduced.
