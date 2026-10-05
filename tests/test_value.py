from cs2form.value import assess


def test_underdog_value():
    r = assess(0.45, 1.4, 3.0, "Fav", "Dog")
    assert r.pick == "b" and "андердога Dog" in r.verdict and 0 < r.stake <= 0.02


def test_no_value_when_model_agrees():
    r = assess(0.615, 1.5, 2.4)
    assert r.pick == "" and r.verdict.startswith("Пропуск")
    assert abs(r.market_a - 0.615) < 0.01


def test_equal_odds_highlight_favorite():
    r = assess(0.66, 1.85, 1.85, "M80", "TYLOO")
    assert r.pick == "a" and "фаворита модели M80" in r.verdict
    r = assess(0.58, 1.85, 1.85, "M80", "TYLOO")
    assert r.pick == "" and "модель выделяет M80" in r.verdict


def test_blend_with_market_and_gap_warning():
    r = assess(0.73, 1.68, 2.15, "Legacy", "1win")  # модель 73%, рынок 56%
    assert abs(r.p_used - (0.73 + r.market_a) / 2) < 1e-9
    assert "расходится с рынком на 17" in r.verdict and r.stake <= 0.01
