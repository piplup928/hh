import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.engine.style.analyzer import analyze
from app.engine.style.preprocess import (binarize_ink, preprocess_upload,
                                         segment_lines, segment_words)
from app.engine.text_layout import chunk_text, wrap_paragraph


def _synthetic_page(lines=5, slanted=False):
    """Draw a synthetic 'handwritten' page with known layout."""
    img = Image.new("L", (900, 700), 245)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(28)
    for i in range(lines):
        y = 60 + i * 100
        x = 40
        for word in ["lorem", "ipsum", "dolor", "sit", "amet"]:
            d.text((x, y), word, fill=30, font=font)
            x += font.getbbox(word)[2] + 30
    return img


def test_preprocess_produces_canvas_and_lines():
    pre = preprocess_upload(_synthetic_page())
    assert pre["style_canvas"].size == (768, 768)
    assert len(pre["line_bands"]) == 5


def test_segment_words_finds_words():
    img = _synthetic_page(lines=1)
    ink = binarize_ink(np.array(img))
    bands = segment_lines(ink)
    assert bands
    y0, y1 = bands[0]
    words = segment_words(ink[y0:y1])
    assert 3 <= len(words) <= 7  # 5 words, tolerate merge/split


def test_analyzer_metrics_sane():
    img = _synthetic_page()
    gray = np.array(img)
    m = analyze(gray, binarize_ink(gray))
    assert m.lines_analyzed == 5
    assert 0.5 < m.stroke_width_px < 20
    assert 60 < m.line_spacing_px < 140  # drawn at 100px pitch
    assert abs(m.slant_deg) < 15         # upright font
    assert 0 < m.ink_darkness <= 1


def test_wrap_and_chunk():
    text = "word " * 500
    lines = wrap_paragraph(text, 42, 8)
    assert len(lines) == 8
    assert all(len(l) <= 42 for l in lines)
    chunks = chunk_text(text, 42, 8)
    assert len(chunks) > 1
    # nothing lost
    assert sum(len(c.split()) for c in chunks) == 500


def test_wrap_breaks_overlong_word():
    lines = wrap_paragraph("a" * 100, 42, 8)
    assert all(len(l) <= 42 for l in lines)
