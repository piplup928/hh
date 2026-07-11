from app.engine.router import latex_normalize, segment


def test_plain_text_single_run():
    runs = segment("Hello world this is prose")
    assert len(runs) == 1
    assert runs[0].kind == "text"


def test_explicit_math_delimiters():
    runs = segment(r"Euler said $e^{i\pi} + 1 = 0$ which is beautiful")
    kinds = [r.kind for r in runs]
    assert kinds == ["text", "math", "text"]
    assert runs[1].content == r"e^{i\pi} + 1 = 0"


def test_bare_numbers_route_to_math():
    runs = segment("The answer is 42 obviously")
    assert [r.kind for r in runs] == ["text", "math", "text"]


def test_scientific_notation_and_chemistry():
    assert segment("6.022e23")[0].kind == "math"
    assert segment("H2O")[0].kind == "math"


def test_unicode_math_symbols():
    runs = segment("area ∑ x2 done")
    assert any(r.kind == "math" for r in runs)


def test_adjacent_math_merges():
    runs = segment("compute 3 + 4 = 7 now")
    math_runs = [r for r in runs if r.kind == "math"]
    assert len(math_runs) == 1
    assert "3" in math_runs[0].content and "7" in math_runs[0].content


def test_latex_normalize_unicode():
    out = latex_normalize("π × ∞ ≤ ∑")
    assert "\\pi" in out and "\\times" in out and "\\infty" in out
    assert "\\leq" in out and "\\sum" in out
