import numpy as np
import pytest
from PIL import Image

from app.benchmarks.style_map import average_precision
from app.engine.ink_ops import embolden, italicize, recolor, scale


def _ink_sample():
    """A diagonal stroke as RGBA ink."""
    arr = np.zeros((60, 120, 4), np.uint8)
    for i in range(50):
        arr[5 + i % 50, 10 + i, 3] = 255
    return Image.fromarray(arr, "RGBA")


def test_recolor_keeps_alpha():
    img = _ink_sample()
    out = recolor(img, "#ff0000")
    a_in = np.array(img)[..., 3]
    a_out = np.array(out)[..., 3]
    assert np.array_equal(a_in, a_out)
    assert np.array(out)[..., 0].max() == 255  # red applied


def test_embolden_increases_ink():
    img = _ink_sample()
    out = embolden(img, stroke_width_px=2.0)
    assert np.array(out)[..., 3].sum() > np.array(img)[..., 3].sum()


def test_italicize_widens():
    img = _ink_sample()
    out = italicize(img, 12)
    assert out.width > img.width
    assert out.height == img.height


def test_scale():
    img = _ink_sample()
    out = scale(img, 1.5)
    assert out.width == int(img.width * 1.5)


def test_average_precision_perfect():
    assert average_precision([True, True, False, False]) == 1.0


def test_average_precision_worst():
    assert average_precision([False, False, True]) == pytest.approx(1 / 3)


def test_average_precision_no_relevant():
    assert average_precision([False, False]) == 0.0


def test_diffink_tokenizer_roundtrip():
    from app.engine.diffink.tokenizer import SOS_ID, EOS_ID, UNK_ID, encode, tokenize

    toks = tokenize(r"\frac{1}{2} + \pi")
    assert "\\frac" in toks and "\\pi" in toks and "{" in toks
    ids = encode(r"\sum_{i=0}^{n} x_i")
    assert ids[0] == SOS_ID and ids[-1] == EOS_ID
    assert UNK_ID not in ids[1:-1]


def test_diffink_stroke_pipeline():
    from app.engine.diffink.render import (normalize_strokes, offsets_to_strokes,
                                           rasterize_strokes, strokes_to_svg)

    rng = np.random.default_rng(0)
    seq = np.column_stack([
        rng.normal(0.5, 0.2, 80), rng.normal(0, 0.3, 80),
        (np.arange(80) % 20 != 0).astype(float)])
    strokes = offsets_to_strokes(seq)
    assert strokes
    strokes = normalize_strokes(strokes, 40)
    svg, (w, h) = strokes_to_svg(strokes, 2.0)
    assert svg.startswith("<svg") and w > 1 and h > 1
    img = rasterize_strokes(strokes, 2.0)
    assert np.array(img)[..., 3].sum() > 0
