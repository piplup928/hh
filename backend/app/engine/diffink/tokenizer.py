"""LaTeX glyph tokenizer for DiffInk conditioning.

Covers the math surface the platform routes to DiffInk: numbers, fractions,
integrals, summations, limits, Greek letters, geometry/set notation, chemical
equations and scientific notation, plus punctuation/symbols.
"""
from __future__ import annotations

import re
from typing import List

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

_GREEK = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau",
    "upsilon", "phi", "chi", "psi", "omega",
]
_COMMANDS = [
    # structure
    "frac", "sqrt", "sum", "prod", "int", "oint", "iint", "lim", "log", "ln",
    "exp", "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos",
    "arctan", "sinh", "cosh", "tanh", "binom", "over", "underset", "overset",
    "vec", "hat", "bar", "dot", "ddot", "tilde", "overline", "underline",
    "left", "right", "begin", "end", "text", "mathrm", "mathbb", "mathcal",
    # relations / operators
    "pm", "mp", "times", "div", "cdot", "ast", "leq", "geq", "neq", "approx",
    "equiv", "sim", "simeq", "propto", "ll", "gg", "subset", "supset",
    "subseteq", "supseteq", "in", "notin", "ni", "cup", "cap", "setminus",
    "emptyset", "forall", "exists", "nabla", "partial", "infty", "perp",
    "parallel", "angle", "triangle", "cong", "degree", "circ", "bullet",
    "rightarrow", "leftarrow", "leftrightarrow", "Rightarrow", "Leftarrow",
    "Leftrightarrow", "mapsto", "to", "uparrow", "downarrow", "rightleftharpoons",
    # dots / misc
    "ldots", "cdots", "vdots", "ddots", "prime", "hbar", "ell", "Re", "Im",
    "aleph", "wp", "otimes", "oplus", "odot", "star", "dagger", "because",
    "therefore", "underbrace", "overbrace", "not",
]
_ASCII = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "+-*/=()[]{}<>|.,;:!?'\"^_~ &%$#@")


def build_vocab() -> List[str]:
    vocab = [PAD, SOS, EOS, UNK]
    vocab += [f"\\{g}" for g in _GREEK]
    vocab += [f"\\{g.capitalize()}" for g in
              ["gamma", "delta", "theta", "lambda", "xi", "pi", "sigma",
               "upsilon", "phi", "psi", "omega"]]
    vocab += [f"\\{c}" for c in _COMMANDS]
    vocab += _ASCII
    return vocab


VOCAB = build_vocab()
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
PAD_ID, SOS_ID, EOS_ID, UNK_ID = (TOKEN_TO_ID[t] for t in (PAD, SOS, EOS, UNK))

_TOKEN_RE = re.compile(r"(\\[A-Za-z]+|\\.|.)", re.DOTALL)


def tokenize(latex: str) -> List[str]:
    out = []
    for m in _TOKEN_RE.finditer(latex.strip()):
        tok = m.group(0)
        if tok.isspace():
            continue
        out.append(tok)
    return out


def encode(latex: str, max_len: int = 256) -> List[int]:
    ids = [SOS_ID]
    for tok in tokenize(latex)[: max_len - 2]:
        ids.append(TOKEN_TO_ID.get(tok, UNK_ID))
    ids.append(EOS_ID)
    return ids


def vocab_size() -> int:
    return len(VOCAB)
