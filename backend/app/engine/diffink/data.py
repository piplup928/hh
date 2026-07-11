"""Online-ink dataset loading for DiffInk training.

Supports InkML corpora: Google MathWriting (recommended, ~400k expressions)
and CROHME. Each sample yields:
  * trajectory (T,3) float32 (dx, dy, pen) normalized
  * LaTeX token ids
  * writer id (when annotated) + style metrics vector computed from the
    rasterized ink (same analyzer as the platform, keeping the conditioning
    distribution consistent between training and inference).
"""
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import STYLE_DIM
from .tokenizer import PAD_ID, encode

NS = {"ink": "http://www.w3.org/2003/InkML"}


def parse_inkml(path: str) -> Optional[Tuple[List[np.ndarray], str, str]]:
    """Returns (strokes[(N,2)], latex, writer)."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    latex, writer = "", ""
    for ann in root.findall("ink:annotation", NS) + root.findall("annotation"):
        t = ann.get("type", "")
        if t in ("truth", "normalizedLabel", "label"):
            latex = latex or (ann.text or "").strip()
        elif t in ("writer", "UI"):
            writer = writer or (ann.text or "").strip()
    strokes = []
    for tr in root.findall("ink:trace", NS) + root.findall("trace"):
        pts = []
        for token in (tr.text or "").strip().split(","):
            vals = token.split()
            if len(vals) >= 2:
                pts.append([float(vals[0]), float(vals[1])])
        if len(pts) >= 2:
            strokes.append(np.array(pts, dtype=np.float64))
    if not strokes or not latex:
        return None
    latex = latex.removeprefix("$").removesuffix("$").strip()
    return strokes, latex, writer


def strokes_to_offsets(strokes: List[np.ndarray], max_len: int, stride: int) -> Optional[np.ndarray]:
    """Absolute strokes -> normalized (T,3) offsets, padded to stride multiple."""
    all_pts = np.concatenate(strokes)
    scale = max(all_pts[:, 1].max() - all_pts[:, 1].min(), 1e-6)
    seq = []
    prev = None
    for s in strokes:
        s = s / scale
        # resample long strokes to bounded density
        if len(s) > 200:
            idx = np.linspace(0, len(s) - 1, 200).astype(int)
            s = s[idx]
        for i, p in enumerate(s):
            if prev is None:
                seq.append([0.0, 0.0, 0.0])
            else:
                pen = 1.0 if i > 0 else 0.0  # first point of stroke = pen-up move
                seq.append([p[0] - prev[0], p[1] - prev[1], pen])
            prev = p
    arr = np.array(seq, dtype=np.float32)
    if len(arr) < 8 or len(arr) > max_len:
        return None
    std = arr[:, :2].std() or 1.0
    arr[:, :2] /= std * 3.0
    pad = (-len(arr)) % stride
    if pad:
        arr = np.concatenate([arr, np.zeros((pad, 3), np.float32)])
    return arr


class InkMLDataset(Dataset):
    def __init__(self, root: str, max_len: int = 2048, stride: int = 4,
                 max_glyphs: int = 128):
        self.files = sorted(
            glob.glob(os.path.join(root, "**", "*.inkml"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"no .inkml files under {root}")
        self.max_len, self.stride, self.max_glyphs = max_len, stride, max_glyphs
        self.writers: dict[str, int] = {}

    def __len__(self):
        return len(self.files)

    def writer_id(self, name: str) -> int:
        if not name:
            return 0
        if name not in self.writers:
            self.writers[name] = (len(self.writers) % 511) + 1
        return self.writers[name]

    def __getitem__(self, i):
        parsed = parse_inkml(self.files[i])
        if parsed is None:
            return self[(i + 1) % len(self)]
        strokes, latex, writer = parsed
        arr = strokes_to_offsets(strokes, self.max_len, self.stride)
        if arr is None:
            return self[(i + 1) % len(self)]
        glyphs = encode(latex, self.max_glyphs)
        style = self._style_vector(arr)
        return (torch.from_numpy(arr), torch.tensor(glyphs, dtype=torch.long),
                torch.from_numpy(style), self.writer_id(writer))

    @staticmethod
    def _style_vector(arr: np.ndarray) -> np.ndarray:
        """Cheap trajectory-level style statistics standing in for raster
        StyleMetrics during training (indices aligned with
        engine.style_metrics_to_vector)."""
        v = np.zeros(STYLE_DIM, dtype=np.float32)
        deltas = arr[arr[:, 2] > 0.5][:, :2]
        if len(deltas) > 4:
            ang = np.arctan2(deltas[:, 1], deltas[:, 0])
            vert = deltas[np.abs(np.abs(ang) - np.pi / 2) < 0.6]
            if len(vert) > 2:
                v[0] = float(np.median(np.arctan2(vert[:, 0], np.abs(vert[:, 1]))) / (np.pi / 4))
            v[3] = float(np.abs(deltas).mean() * 4)
            v[11] = float((arr[:, 2] > 0.5).mean())
        return v


def collate(batch):
    trajs, glyphs, styles, writers = zip(*batch)
    T = max(t.shape[0] for t in trajs)
    G = max(g.shape[0] for g in glyphs)
    tb = torch.zeros(len(batch), T, 3)
    gb = torch.full((len(batch), G), PAD_ID, dtype=torch.long)
    for i, (t, g) in enumerate(zip(trajs, glyphs)):
        tb[i, : t.shape[0]] = t
        gb[i, : g.shape[0]] = g
    return tb, gb, torch.stack(styles), torch.tensor(writers, dtype=torch.long)
