"""Tests du cœur du projet : comptabilité du portefeuille, calendrier, optimiseurs des baselines,
incertitude statistique, verrou du test final et registre des agents."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    estimation_window,
    lowvol_weights,
    max_sharpe_weights,
    min_variance_weights,
    momentum_weights,
    rebalance_flags,
    risk_parity_weights,
)
from src.env import initial_weights, rebalance_step
from src.evaluate import check_final_test_lock, metrics_table, to_markdown
from src.features import split_transitions, weekly_asset_returns, weekly_decision_dates
from src.metrics import block_bootstrap_indices, sharpe_confidence_interval, sharpe_difference_test, sharpe_ratio
from src.train import ALGORITHMS, get_factory

P = 52


# ------------------------------------------------------------------ comptabilité
def test_rebalance_step_accounting() -> None:
    r = np.array([0.05, -0.03, 0.001])
    hold_cash = rebalance_step(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), r, 0.001, include_cash=False)
    assert hold_cash.turnover == 0 and np.isclose(hold_cash.net_return, 0.001)
    invest = rebalance_step(np.array([0.0, 0.0, 1.0]), np.array([0.5, 0.5, 0.0]), r, 0.001, include_cash=False)
    assert np.isclose(invest.turnover, 1.0) and np.isclose(invest.cost, 0.001)  # jambe cash gratuite
    assert np.isclose(invest.net_return, 0.5 * 0.05 - 0.5 * 0.03 - 0.001)
    np.testing.assert_allclose(invest.drifted_weights, np.array([0.5 * 1.05, 0.5 * 0.97, 0.0]) / (1 + 0.01))
    swap = rebalance_step(np.array([0.5, 0.5, 0.0]), np.array([0.4, 0.6, 0.0]), np.zeros(3), 0.001, include_cash=False)
    assert np.isclose(swap.turnover, 0.2)
    np.testing.assert_allclose(initial_weights(5, "equal_risky"), [0.25, 0.25, 0.25, 0.25, 0.0])


def test_weekly_calendar_and_splits() -> None:
    idx = pd.bdate_range("2021-01-04", "2021-03-26").drop(pd.Timestamp("2021-01-15"))  # vendredi férié
    dates = weekly_decision_dates(idx, "W-FRI")
    assert pd.Timestamp("2021-01-14") in dates and pd.Timestamp("2021-01-15") not in dates
    prices = pd.DataFrame({"A": np.linspace(10, 20, len(idx))}, idx)
    wr = weekly_asset_returns(prices, dates)
    daily = prices["A"].pct_change()
    in_week = daily.loc[(daily.index > dates[2]) & (daily.index <= dates[3])]
    assert np.isclose(wr.iloc[2, 0], np.prod(1 + in_week) - 1)
    tr = split_transitions(dates, "2021-01-01", "2021-02-12")
    assert dates[tr[-1] + 1] <= pd.Timestamp("2021-02-12")  # période de détention incluse dans le split


# ------------------------------------------------------------------ optimiseurs des baselines
def test_optimizers() -> None:
    var = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(min_variance_weights(np.diag(var), "CLARABEL"), (1 / var) / (1 / var).sum(), atol=1e-6)
    vol = np.sqrt(var)
    np.testing.assert_allclose(risk_parity_weights(np.diag(var), "CLARABEL"), (1 / vol) / (1 / vol).sum(), atol=1e-5)
    np.testing.assert_allclose(max_sharpe_weights(np.array([0.1, 0.2, 0.3]), np.eye(3), 0.0, 1.0, "CLARABEL"),
                               [1 / 6, 2 / 6, 3 / 6], atol=1e-5)
    rng = np.random.default_rng(0)
    cov = np.cov((rng.normal(size=(500, 10)) * 0.01).T)
    w = max_sharpe_weights(np.linspace(-2e-4, 1e-3, 10), cov, 0.0, 0.15, "CLARABEL")
    assert np.isclose(w.sum(), 1) and (w >= 0).all() and w.max() <= 0.15 + 1e-6
    w_few = max_sharpe_weights(np.array([1e-3, 2e-3, -1e-3]), np.eye(3) * 1e-4, 0.0, 0.15, "CLARABEL")
    assert np.isclose(w_few.sum(), 1)  # plafond relâché quand il est infaisable


def test_signal_baselines() -> None:
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.DataFrame(0.0, idx, list("ABCDEF"))
    r.loc[idx[-252:-21], "C"] = 0.002
    r.loc[idx[-252:-21], "A"] = 0.001
    r.loc[idx[-21:], "F"] = 0.05  # dernier mois ignoré par le momentum 12-1
    w = momentum_weights(r, 252, 21, 2)
    assert w[2] == 0.5 and w[0] == 0.5 and w[5] == 0.0
    r2 = pd.DataFrame(np.random.default_rng(1).normal(0, 1, (60, 4)) * np.array([0.01, 0.03, 0.005, 0.02]), columns=list("ABCD"))
    np.testing.assert_allclose(lowvol_weights(r2, 2), [0.5, 0.0, 0.5, 0.0])
    h = estimation_window(pd.DataFrame({"a": np.arange(100.0)}, pd.bdate_range("2020-01-01", periods=100)),
                          pd.bdate_range("2020-01-01", periods=100)[50], 30, 10)
    assert len(h) == 30 and h["a"].iloc[-1] == 50
    dates = pd.date_range("2021-01-01", periods=20, freq="W-FRI")
    assert len(rebalance_flags(dates, np.arange(19), "monthly")) == len({d.to_period("M") for d in dates[:19]})


# ------------------------------------------------------------------ incertitude
def test_bootstrap() -> None:
    idx = block_bootstrap_indices(10, 4, 50, np.random.default_rng(0))
    assert idx.shape == (50, 10) and ((idx[:, 1] - idx[:, 0]) % 10 == 1).all()
    r = np.random.default_rng(1).normal(0.003, 0.02, 520)
    lo, hi = sharpe_confidence_interval(r, P, 0.0, 2000, 4, 0.95, 7)
    assert lo < sharpe_ratio(r, P) < hi
    base = np.random.default_rng(2).normal(0.001, 0.02, 300)
    assert sharpe_difference_test(base, base, P, 0.0, 1000, 4, 0)["p_value"] == 1.0
    assert sharpe_difference_test(base + 0.004, base, P, 0.0, 1000, 4, 0)["p_value"] < 0.01


# ------------------------------------------------------------------ évaluation
def test_final_test_lock(tmp_path) -> None:
    lock = tmp_path / ".final_test_done"
    check_final_test_lock(lock, force=False)
    lock.write_text('{"run_at": "x"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="REFUS"):
        check_final_test_lock(lock, force=False)
    check_final_test_lock(lock, force=True)


def test_tables_and_registry(cfg: dict) -> None:
    df = pd.DataFrame({"net_return": [0.01, -0.02, 0.03], "turnover": [1.0, 0.1, 0.2]})
    tab = metrics_table({"A": df, "B": df * 0.5}, cfg, uncertainty=False)
    assert list(tab.index) == ["A", "B"] and "sharpe" in to_markdown(tab).splitlines()[0].lower()
    assert set(ALGORITHMS) == {"PPO", "A2C", "SAC", "TD3"} and get_factory(cfg).name == "PPO"
    for name in ("A2C", "SAC", "TD3"):
        with pytest.raises(NotImplementedError):
            ALGORITHMS[name](cfg).build(None, 0, None)
