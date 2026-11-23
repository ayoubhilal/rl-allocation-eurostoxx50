"""Phase 5 : métriques de performance sur des rendements périodiques (hebdomadaires par défaut)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series | list[float]


def _arr(r: ArrayLike) -> np.ndarray:
    a = np.asarray(r, dtype=float)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError("série de rendements 1D non vide attendue")
    return a


def periodic_rf(annual_rf: float, periods_per_year: int) -> float:
    return (1.0 + annual_rf) ** (1.0 / periods_per_year) - 1.0


def cumulative_return(r: ArrayLike) -> float:
    return float(np.prod(1.0 + _arr(r)) - 1.0)


def annualized_return(r: ArrayLike, periods_per_year: int) -> float:
    """Rendement géométrique annualisé."""
    a = _arr(r)
    growth = np.prod(1.0 + a)
    return float(growth ** (periods_per_year / len(a)) - 1.0) if growth > 0 else -1.0


def annualized_volatility(r: ArrayLike, periods_per_year: int) -> float:
    a = _arr(r)
    return float(a.std(ddof=1) * np.sqrt(periods_per_year)) if len(a) > 1 else float("nan")


def sharpe_ratio(r: ArrayLike, periods_per_year: int, annual_rf: float = 0.0) -> float:
    """Sharpe annualisé : moyenne / écart-type des rendements excédentaires x sqrt(périodes)."""
    ex = _arr(r) - periodic_rf(annual_rf, periods_per_year)
    sd = ex.std(ddof=1) if len(ex) > 1 else 0.0
    return float(ex.mean() / sd * np.sqrt(periods_per_year)) if sd > 1e-15 else float("nan")


def sortino_ratio(r: ArrayLike, periods_per_year: int, annual_rf: float = 0.0) -> float:
    """Sortino annualisé ; downside deviation = sqrt(moyenne(min(r - rf, 0)^2))."""
    ex = _arr(r) - periodic_rf(annual_rf, periods_per_year)
    dd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    if dd < 1e-15:
        return float("inf") if ex.mean() > 0 else float("nan")
    return float(ex.mean() / dd * np.sqrt(periods_per_year))


def drawdown_series(r: ArrayLike) -> np.ndarray:
    """Drawdown à chaque date (valeur initiale 1 incluse dans le plus-haut courant)."""
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + _arr(r))])
    return (wealth / np.maximum.accumulate(wealth) - 1.0)[1:]


def max_drawdown(r: ArrayLike) -> float:
    """Perte maximale depuis un plus-haut (nombre <= 0)."""
    return float(min(drawdown_series(r).min(), 0.0))


def calmar_ratio(r: ArrayLike, periods_per_year: int) -> float:
    mdd = max_drawdown(r)
    return float(annualized_return(r, periods_per_year) / abs(mdd)) if mdd < -1e-15 else float("nan")


def summarize(
    returns: ArrayLike, turnover: ArrayLike | None, periods_per_year: int, annual_rf: float = 0.0
) -> dict[str, Any]:
    """Toutes les métriques du projet pour une stratégie."""
    return {
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_vol": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year, annual_rf),
        "sortino": sortino_ratio(returns, periods_per_year, annual_rf),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "avg_turnover": float(np.mean(_arr(turnover))) if turnover is not None else float("nan"),
        "cum_return": cumulative_return(returns),
        "n_periods": len(_arr(returns)),
    }


def metrics_from_frame(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Métriques à partir d'un DataFrame de backtest (colonnes net_return, turnover)."""
    m = cfg["metrics"]
    return summarize(df["net_return"], df["turnover"], m["periods_per_year"], m["annual_risk_free"])


# --------------------------------------------------------------------------- incertitude (bootstrap)
def block_bootstrap_indices(n: int, block_length: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Indices de bootstrap par blocs circulaires (préserve l'autocorrélation à court terme) : (n_resamples, n)."""
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_length)[None, None, :]) % n
    return idx.reshape(n_resamples, -1)[:, :n]


def _sharpe_rows(x: np.ndarray, periods_per_year: int, rf_p: float) -> np.ndarray:
    ex = x - rf_p
    sd = ex.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 1e-15, ex.mean(axis=1) / sd * np.sqrt(periods_per_year), np.nan)


def sharpe_confidence_interval(r: ArrayLike, periods_per_year: int, annual_rf: float, n_resamples: int,
                               block_length: int, confidence: float, seed: int) -> tuple[float, float]:
    """Intervalle de confiance (percentile) du Sharpe annualisé par bootstrap par blocs."""
    a = _arr(r)
    idx = block_bootstrap_indices(len(a), block_length, n_resamples, np.random.default_rng(seed))
    s = _sharpe_rows(a[idx], periods_per_year, periodic_rf(annual_rf, periods_per_year))
    q = (1 - confidence) / 2
    return float(np.nanquantile(s, q)), float(np.nanquantile(s, 1 - q))


def sharpe_difference_test(r_a: ArrayLike, r_b: ArrayLike, periods_per_year: int, annual_rf: float,
                           n_resamples: int, block_length: int, seed: int) -> dict[str, float]:
    """Différence de Sharpe (a - b) sur les MÊMES semaines ; p-value bilatérale par bootstrap apparié
    (distribution recentrée sous H0 : différence nulle)."""
    a, b = _arr(r_a), _arr(r_b)
    if len(a) != len(b):
        raise ValueError("séries de longueurs différentes")
    rf_p = periodic_rf(annual_rf, periods_per_year)
    diff = sharpe_ratio(a, periods_per_year, annual_rf) - sharpe_ratio(b, periods_per_year, annual_rf)
    idx = block_bootstrap_indices(len(a), block_length, n_resamples, np.random.default_rng(seed))
    d = _sharpe_rows(a[idx], periods_per_year, rf_p) - _sharpe_rows(b[idx], periods_per_year, rf_p)
    d = d[np.isfinite(d)]
    if not np.isfinite(diff) or len(d) == 0 or np.allclose(a, b):
        return {"diff": float(diff) if np.isfinite(diff) else float("nan"), "p_value": 1.0}
    p = float(np.mean(np.abs(d - d.mean()) >= abs(diff)))
    return {"diff": float(diff), "p_value": max(p, 1.0 / len(d))}


def uncertainty_columns(returns: dict[str, ArrayLike], cfg: dict[str, Any]) -> pd.DataFrame:
    """IC du Sharpe de chaque stratégie + test de différence contre la stratégie de référence (config)."""
    m, bs = cfg["metrics"], cfg["metrics"]["bootstrap"]
    ref = returns.get(bs["reference"])
    rows = {}
    for name, r in returns.items():
        lo, hi = sharpe_confidence_interval(r, m["periods_per_year"], m["annual_risk_free"], bs["n_resamples"],
                                            bs["block_length"], bs["confidence"], bs["seed"])
        p = float("nan")
        if ref is not None and name != bs["reference"]:
            p = sharpe_difference_test(r, ref, m["periods_per_year"], m["annual_risk_free"], bs["n_resamples"],
                                       bs["block_length"], bs["seed"])["p_value"]
        rows[name] = {"sharpe_ci_low": lo, "sharpe_ci_high": hi, "p_vs_ref": p}
    return pd.DataFrame(rows).T
