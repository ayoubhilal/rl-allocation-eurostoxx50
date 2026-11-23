"""Tests Phase 5 : métriques sur séries synthétiques à valeur connue."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cumulative_return,
    drawdown_series,
    max_drawdown,
    periodic_rf,
    sharpe_ratio,
    sortino_ratio,
    summarize,
)

P = 52


def test_constant_returns() -> None:
    r = np.full(52, 0.01)
    assert np.isclose(cumulative_return(r), 1.01 ** 52 - 1)
    assert np.isclose(annualized_return(r, P), 1.01 ** 52 - 1)
    assert np.isclose(annualized_volatility(r, P), 0.0)
    assert np.isnan(sharpe_ratio(r, P))          # écart-type nul
    assert sortino_ratio(r, P) == float("inf")   # aucune perte
    assert max_drawdown(r) == 0.0 and np.isnan(calmar_ratio(r, P))


def test_annualization_two_years() -> None:
    r = np.full(104, (1.1) ** (1 / 52) - 1)  # +10 %/an pendant 2 ans
    assert np.isclose(annualized_return(r, P), 0.10)
    assert np.isclose(cumulative_return(r), 0.21)


def test_sharpe_and_vol_alternating() -> None:
    r = np.array([0.02, -0.01] * 26)
    mu, sd = r.mean(), r.std(ddof=1)
    assert np.isclose(annualized_volatility(r, P), sd * np.sqrt(P))
    assert np.isclose(sharpe_ratio(r, P), mu / sd * np.sqrt(P))
    rf = 0.02
    assert np.isclose(sharpe_ratio(r, P, rf), (mu - periodic_rf(rf, P)) / sd * np.sqrt(P))
    down = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert np.isclose(sortino_ratio(r, P), mu / down * np.sqrt(P))
    assert sortino_ratio(r, P) > sharpe_ratio(r, P)


def test_drawdown_known_path() -> None:
    r = np.array([0.10, -0.50, 0.20, 0.50, -0.10])
    # richesse : 1.1, 0.55, 0.66, 0.99, 0.891 ; plus-haut 1.1
    assert np.isclose(max_drawdown(r), 0.55 / 1.1 - 1)
    np.testing.assert_allclose(drawdown_series(r), [0, -0.5, 0.66 / 1.1 - 1, 0.99 / 1.1 - 1, 0.891 / 1.1 - 1])
    assert np.isclose(calmar_ratio(r, P), annualized_return(r, P) / 0.5)
    assert np.isclose(max_drawdown([-0.2, 0.1]), -0.2)  # perte dès la 1re période (plus-haut initial = 1)


def test_summarize_keys_and_turnover() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.002, 0.02, 200)
    s = summarize(r, np.full(200, 0.1), P)
    assert set(s) == {"ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "avg_turnover",
                      "cum_return", "n_periods"}
    assert np.isclose(s["avg_turnover"], 0.1) and s["n_periods"] == 200
    assert -1 <= s["max_drawdown"] <= 0


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        annualized_return([], P)
