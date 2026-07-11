"""Text chunking/wrapping for the Paragraph-LDM canvas.

The released model generates fixed 768x768 paragraph pages with a bounded
number of lines/characters. Long documents are split into paragraph chunks;
each chunk is wrapped to the model's line capacity. Chunk boundaries land on
sentence/word boundaries so the imitation stays natural.
"""
from __future__ import annotations

from typing import List


def wrap_paragraph(text: str, max_chars: int, max_lines: int) -> List[str]:
    """Greedy word wrap honoring explicit newlines; truncates at max_lines
    (callers chunk beforehand via chunk_text, so truncation is a safety net)."""
    lines: List[str] = []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        cur = ""
        for w in words:
            # hard-break single words longer than a line
            while len(w) > max_chars:
                if cur:
                    lines.append(cur)
                    cur = ""
                lines.append(w[:max_chars])
                w = w[max_chars:]
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_chars:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
    while lines and lines[-1] == "":
        lines.pop()
    return lines[:max_lines]


def chunk_text(text: str, max_chars: int, max_lines: int) -> List[str]:
    """Split arbitrary-length text into chunks that each fit one LDM page."""
    budget = max_chars * max_lines
    chunks: List[str] = []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            continue
        cur_words: List[str] = []
        cur_len = 0
        for w in words:
            add = len(w) + (1 if cur_words else 0)
            if cur_len + add > budget * 0.92 and cur_words:
                chunks.append(" ".join(cur_words))
                cur_words, cur_len = [w], len(w)
            else:
                cur_words.append(w)
                cur_len += add
        if cur_words:
            chunks.append(" ".join(cur_words))
    return chunks or [""]
