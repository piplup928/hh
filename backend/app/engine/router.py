"""Content router: splits editor text into engine-typed runs.

Routing policy (per product spec):
  * prose / alphabetic words             -> Paragraph-LDM
  * explicit math ($...$, \\(...\\), ``\\command``) -> DiffInk
  * bare numbers, scientific notation, math/geometry/chemistry symbols,
    and standalone punctuation runs      -> DiffInk

Adjacent same-engine runs merge so the LDM sees whole paragraphs (its
strength: long-range spacing + baseline continuity) and DiffInk sees whole
expressions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal

RunKind = Literal["text", "math"]

# characters that force the math engine
MATH_CHARS = set("0123456789=+−–×÷±<>≤≥≠≈∞∫∑∏√∂∇πθλμσΩαβγδεφψω∈∉⊂⊃∪∩∅∀∃→←↔⇒⇐⇔°′″%‰^_|~")
_MATH_INLINE = re.compile(r"(\$[^$]+\$|\\\([^)]*\\\)|\\\[[^\]]*\\\])")
_NUMERIC = re.compile(r"^[0-9]+([.,][0-9]+)?([eE][+-]?[0-9]+)?$")
_CHEM = re.compile(r"^([A-Z][a-z]?[0-9]*){2,}$")  # H2O, CO2, NaCl...
_PUNCT_RUN = re.compile(r"^[\W_]+$", re.UNICODE)


@dataclass
class Run:
    kind: RunKind
    content: str  # raw text, or LaTeX body for math


def _classify_word(w: str) -> RunKind:
    if not w:
        return "text"
    if _NUMERIC.match(w) or _CHEM.match(w):
        return "math"
    mathy = sum(1 for ch in w if ch in MATH_CHARS)
    if mathy and mathy >= max(len(w) // 2, 1):
        return "math"
    if _PUNCT_RUN.match(w) and any(ch in MATH_CHARS for ch in w):
        return "math"
    return "text"


def segment(text: str) -> List[Run]:
    """Split text into engine runs. Explicit $...$ delimiters win; the rest is
    classified word-by-word, then merged."""
    runs: List[Run] = []

    def push(kind: RunKind, content: str):
        if not content:
            return
        if runs and runs[-1].kind == kind:
            joiner = "" if kind == "math" and runs[-1].content.endswith(" ") else " "
            runs[-1].content = (runs[-1].content + joiner + content).strip()
        else:
            runs.append(Run(kind, content.strip()))

    pos = 0
    for m in _MATH_INLINE.finditer(text):
        before = text[pos:m.start()]
        _segment_plain(before, push)
        body = m.group(0)
        if body.startswith("$"):
            body = body.strip("$")
        else:
            body = body[2:-2]
        push("math", body.strip())
        pos = m.end()
    _segment_plain(text[pos:], push)
    return [r for r in runs if r.content]


def _segment_plain(chunk: str, push):
    for word in chunk.split():
        push(_classify_word(word), word)


def latex_normalize(math_text: str) -> str:
    """Map common unicode math input to LaTeX for DiffInk conditioning."""
    table = {
        "×": r"\times ", "÷": r"\div ", "±": r"\pm ", "≤": r"\leq ", "≥": r"\geq ",
        "≠": r"\neq ", "≈": r"\approx ", "∞": r"\infty ", "∫": r"\int ",
        "∑": r"\sum ", "∏": r"\prod ", "√": r"\sqrt ", "∂": r"\partial ",
        "∇": r"\nabla ", "π": r"\pi ", "θ": r"\theta ", "λ": r"\lambda ",
        "μ": r"\mu ", "σ": r"\sigma ", "Ω": r"\Omega ", "α": r"\alpha ",
        "β": r"\beta ", "γ": r"\gamma ", "δ": r"\delta ", "ε": r"\epsilon ",
        "φ": r"\phi ", "ψ": r"\psi ", "ω": r"\omega ", "∈": r"\in ",
        "∉": r"\notin ", "⊂": r"\subset ", "∪": r"\cup ", "∩": r"\cap ",
        "∅": r"\emptyset ", "∀": r"\forall ", "∃": r"\exists ",
        "→": r"\rightarrow ", "←": r"\leftarrow ", "⇒": r"\Rightarrow ",
        "⇔": r"\Leftrightarrow ", "°": r"\degree ", "−": "-", "–": "-",
    }
    out = math_text
    for k, v in table.items():
        out = out.replace(k, v)
    return re.sub(r"\s+", " ", out).strip()
