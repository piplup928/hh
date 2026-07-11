"""Unit tests for the official-DiffInk adapter that don't require weights."""
import numpy as np
import pytest

from app.engine.diffink.engine import DiffInkEngine
from app.engine.diffink.official import OfficialDiffInk


def test_latex_to_plain():
    f = DiffInkEngine.latex_to_plain
    assert f(r"\frac{1}{2}") == "1/2"
    out = f(r"\int_0^1 x \cdot \pi \, dx \leq \infty")
    assert "∫" in out and "π" in out and "≤" in out and "∞" in out
    assert "\\" not in out and "{" not in out


def test_points_to_offsets_shape_and_pen():
    pts = np.zeros((10, 5), np.float32)
    pts[:, 0] = 1.0
    pts[:5, 2] = 1.0  # pen down first half
    out = OfficialDiffInk.points_to_offsets(pts)
    assert out.shape == (10, 3)
    assert out[:5, 2].all() and not out[5:, 2].any()


def test_encode_text_uses_dict_order_and_terminator():
    ad = OfficialDiffInk()
    ad._chars = {c: i for i, c in enumerate("abc12、")}
    ids, skipped = ad.encode_text("ab1Ω")
    assert skipped == ["Ω"]
    seq = ids[0].tolist()
    assert seq[:3] == [0, 1, 3]           # dict-order ids
    assert seq[-1] == ad._chars["、"]      # terminator appended


def test_encode_text_all_unknown_raises():
    ad = OfficialDiffInk()
    ad._chars = {"a": 0}
    with pytest.raises(ValueError):
        ad.encode_text("ΩΨΦ")


def test_engine_status_reports_both_backends():
    st = DiffInkEngine().status()
    assert "official" in st and "scratch" in st
    assert "backend" in st["official"]
