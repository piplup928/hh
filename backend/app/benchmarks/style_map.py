"""Style Preservation Score — writer-retrieval mean Average Precision (mAP).

Standard protocol from the writer identification/retrieval literature
(as used to evaluate styled HTG, e.g. VATr / HWD papers):

  * Gallery: real handwriting lines from N writers — the user's uploaded
    reference lines plus distractor writers (bundled IAM paragraphs from the
    Paragraph-LDM repo, segmented into lines).
  * Queries: generated handwriting images (model output for the user style).
  * Each query ranks the gallery by feature distance; AP is computed with the
    user's real lines as the relevant set; mAP averages over queries.

Features come from the official HWD VGG16 backbone (trained on the font_square
synthetic handwriting corpus — the same features the HWD score uses), so the
score measures style proximity, not content similarity.
"""
from __future__ import annotations

from typing import Dict, List

import torch


def image_level_features(processed) -> Dict[int, tuple]:
    """HWD ProcessedDataset (per-column tokens) -> per-image mean feature."""
    feats: Dict[int, list] = {}
    authors: Dict[int, str] = {}
    ids = processed.ids.tolist()
    for row, (img_id, author) in enumerate(zip(ids, processed.authors)):
        feats.setdefault(int(img_id), []).append(processed.features[row])
        authors[int(img_id)] = author
    out = {}
    for img_id, rows in feats.items():
        f = torch.stack(rows).mean(dim=0)
        out[img_id] = (torch.nn.functional.normalize(f, dim=0), authors[img_id])
    return out


def average_precision(ranked_relevance: List[bool]) -> float:
    hits, precisions = 0, []
    for i, rel in enumerate(ranked_relevance, start=1):
        if rel:
            hits += 1
            precisions.append(hits / i)
    return float(sum(precisions) / len(precisions)) if precisions else 0.0


def style_map(query_processed, gallery_processed, target_author: str) -> dict:
    """Compute writer-retrieval mAP of generated queries against a real gallery."""
    queries = image_level_features(query_processed)
    gallery = image_level_features(gallery_processed)
    if not queries or not gallery:
        raise ValueError("empty query or gallery features")

    g_feats = torch.stack([f for f, _ in gallery.values()])
    g_authors = [a for _, a in gallery.values()]
    n_relevant = sum(1 for a in g_authors if a == target_author)
    if n_relevant == 0:
        raise ValueError(f"gallery contains no lines for author {target_author!r}")

    aps, top1_hits = [], 0
    for f, _ in queries.values():
        d = torch.cdist(f.unsqueeze(0), g_feats).squeeze(0)  # euclidean, HWD-style
        order = torch.argsort(d)
        ranked = [g_authors[i] == target_author for i in order.tolist()]
        aps.append(average_precision(ranked))
        if ranked[0]:
            top1_hits += 1

    return {
        "mAP": float(sum(aps) / len(aps)),
        "top1": top1_hits / len(aps),
        "nQueries": len(aps),
        "nGallery": len(g_authors),
        "nRelevant": n_relevant,
        "nDistractorAuthors": len(set(g_authors)) - 1,
    }
