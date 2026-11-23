"""Phase 4 : stratégies classiques évaluées avec la même comptabilité hebdomadaire que l'agent RL."""

from __future__ import annotations

import logging
from typing import Any, Callable

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.env import initial_weights, rebalance_step, results_frame
from src.features import MarketData

logger = logging.getLogger(__name__)

# target_fn(k, poids dérivés) -> poids cibles (N+1) ou None (pas de transaction)
TargetFn = Callable[[int, np.ndarray], np.ndarray | None]


def simulate(
    weekly_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    idx: np.ndarray,
    target_fn: TargetFn,
    assets: list[str],
    cost_rate: float,
    init_mode: str,
    include_cash: bool = True,
    available: np.ndarray | None = None,
) -> pd.DataFrame:
    """Backtest hebdomadaire générique (mêmes coûts, mêmes dates et même dérive que l'environnement).

    available (K, N) : titres investissables (univers dynamique) ; le départ « equal_risky » ne porte que sur eux.
    """
    w = initial_weights(len(assets), init_mode)
    if available is not None and init_mode == "equal_risky":
        m = available[int(idx[0])]
        w = np.append(m / m.sum(), 0.0)
    value, rows = 1.0, []
    for k in idx:
        target = target_fn(int(k), w)
        w_new = w.copy() if target is None else np.asarray(target, dtype=float)
        if not np.isclose(w_new.sum(), 1.0) or (w_new < -1e-12).any():
            raise ValueError(f"poids invalides à {dates[k]} : somme={w_new.sum()}")
        res = rebalance_step(w, w_new, weekly_returns[k], cost_rate, include_cash)
        value *= 1.0 + res.net_return
        rows.append({"date": dates[k], "end_date": dates[k + 1], "weights": w_new, "turnover": res.turnover,
                     "cost": res.cost, "gross_return": res.gross_return, "net_return": res.net_return,
                     "value": value})
        w = res.drifted_weights
    return results_frame(rows, assets)


def rebalance_flags(dates: pd.DatetimeIndex, idx: np.ndarray, frequency: str) -> set[int]:
    """Indices de rebalancement : chaque semaine, ou 1re décision de chaque mois (et 1re du split)."""
    if frequency == "weekly":
        return set(int(k) for k in idx)
    if frequency == "monthly":
        flags = {int(idx[0])}
        flags |= {int(k) for k in idx[1:] if dates[k].to_period("M") != dates[k - 1].to_period("M")}
        return flags
    raise ValueError(f"fréquence inconnue : {frequency}")


# --------------------------------------------------------------------------- optimiseurs
def estimation_window(daily_returns: pd.DataFrame, date: pd.Timestamp, window: int, min_obs: int) -> pd.DataFrame | None:
    """Derniers `window` rendements journaliers <= date (aucune donnée future)."""
    hist = daily_returns.loc[:date].iloc[-window:]
    return hist if len(hist) >= min_obs else None


def ledoit_wolf_cov(hist: pd.DataFrame) -> np.ndarray:
    return LedoitWolf().fit(hist.to_numpy()).covariance_


def _clean(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float).ravel(), 0.0, None)
    return w / w.sum()


def min_variance_weights(cov: np.ndarray, solver: str, max_weight: float | None = None) -> np.ndarray:
    """min w' S w  s.c.  sum w = 1, w >= 0 (et w <= max_weight si fourni)."""
    n = cov.shape[0]
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0]
    if max_weight is not None:
        cons.append(w <= max(max_weight, 1.0 / n + 1e-9))
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), cons).solve(solver=solver)
    if w.value is None:
        raise RuntimeError("échec de l'optimisation min-variance")
    return _clean(w.value)


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, rf: float, max_weight: float, solver: str) -> np.ndarray:
    """Tangente long-only plafonnée, reformulation convexe (y = w / k) :
    min y' S y  s.c.  (mu - rf)' y = 1, y >= 0, y <= max_weight * sum(y).
    Si aucun rendement excédentaire n'est positif : repli sur la variance minimale plafonnée.
    """
    ex = mu - rf
    max_weight = max(max_weight, 1.0 / len(mu) + 1e-9)  # plafond infaisable si trop peu de titres éligibles
    if ex.max() <= 0:
        logger.warning("Aucun rendement excédentaire positif : repli max-Sharpe -> min-variance plafonnée")
        return min_variance_weights(cov, solver, max_weight)
    y = cp.Variable(len(mu))
    cons = [ex @ y == 1, y >= 0, y <= max_weight * cp.sum(y)]
    cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov))), cons).solve(solver=solver)
    if y.value is None:  # infaisable : aucun portefeuille plafonné n'a de rendement excédentaire positif
        logger.warning("max-Sharpe infaisable sous le plafond : repli sur la min-variance plafonnée")
        return min_variance_weights(cov, solver, max_weight)
    return _clean(y.value)


def risk_parity_weights(cov: np.ndarray, solver: str) -> np.ndarray:
    """Parité de risque (contributions au risque égales), via le problème convexe
    min 1/2 y' S y - (1/n) sum log y, puis w = y / sum y (Spinu, 2013)."""
    n = cov.shape[0]
    y = cp.Variable(n, pos=True)
    cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(cov)) - cp.sum(cp.log(y)) / n)).solve(solver=solver)
    if y.value is None:
        raise RuntimeError("échec de l'optimisation parité de risque")
    return _clean(y.value)


def momentum_weights(hist: pd.DataFrame, lookback: int, skip: int, top_k: int) -> np.ndarray:
    """Momentum cross-sectionnel « 12-1 » : rendement cumulé sur `lookback` jours hors les `skip` derniers,
    top-k titres équipondérés (hist = rendements journaliers <= t)."""
    window = hist.iloc[-lookback:-skip] if skip > 0 else hist.iloc[-lookback:]
    score = (1.0 + window).prod(axis=0) - 1.0
    top = np.argsort(-score.to_numpy(), kind="mergesort")[:top_k]
    w = np.zeros(hist.shape[1])
    w[top] = 1.0 / top_k
    return w


def lowvol_weights(hist: pd.DataFrame, top_k: int) -> np.ndarray:
    """Faible volatilité : les top-k titres les moins volatils sur la fenêtre, équipondérés."""
    vol = hist.std(axis=0).to_numpy()
    low = np.argsort(vol, kind="mergesort")[:top_k]
    w = np.zeros(hist.shape[1])
    w[low] = 1.0 / top_k
    return w


# --------------------------------------------------------------------------- stratégies
def _sim(md: MarketData, split: str, cfg: dict[str, Any], weekly_returns: np.ndarray, target_fn: TargetFn,
         assets: list[str]) -> pd.DataFrame:
    e = cfg["env"]
    dynamic = md.available if weekly_returns.shape[1] == md.n_assets + 1 else None
    return simulate(weekly_returns, md.dates, md.splits[split], target_fn, assets, e["transaction_cost"],
                    e["eval_initial_allocation"], e["turnover_includes_cash"], dynamic)


def benchmark_label(ticker: str) -> str:
    return ticker.lstrip("^").split(".")[0]


def run_buy_and_hold(md: MarketData, split: str, cfg: dict[str, Any], j: int = 0) -> pd.DataFrame:
    """Achat du benchmark j au 1er pas (depuis la même allocation initiale), puis détention."""
    wr = np.column_stack([md.benchmark_weekly[:, j], md.weekly_returns[:, -1]])
    start = int(md.splits[split][0])
    fn: TargetFn = lambda k, w: np.array([1.0, 0.0]) if k == start else None  # noqa: E731
    return _sim(md, split, cfg, wr, fn, [md.benchmark_names[j], md.cash_name])


def run_equal_weight(md: MarketData, split: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """1/N sur les titres disponibles, rebalancé chaque semaine."""
    def target(k: int, w: np.ndarray) -> np.ndarray:
        m = np.ones(md.n_assets, dtype=bool) if md.available is None else md.available[k]
        return np.append(m / m.sum(), 0.0)

    return _sim(md, split, cfg, md.weekly_returns, target, md.all_assets)


def periodic_target_fn(md: MarketData, idx: np.ndarray, cfg: dict[str, Any], kind: str) -> TargetFn:
    """Cible de la stratégie `kind` aux dates de rebalancement de idx (None ailleurs), données <= t."""
    bcfg = cfg["baselines"]
    flags = rebalance_flags(md.dates, idx, bcfg["rebalance_frequency"])
    risky_daily = md.daily_returns[md.assets]
    rf_daily = (1 + bcfg["annual_risk_free"]) ** (1 / cfg["features"]["annualization_days"]) - 1

    def eligible(hist: pd.DataFrame, k: int) -> pd.DataFrame:
        """Titres disponibles en k ET sans trou sur toute la fenêtre d'estimation."""
        cols = hist.columns[hist.notna().all().to_numpy()]
        if md.available is not None:
            cols = [c for c in cols if md.available[k][md.assets.index(c)]]
        return hist[cols]

    def expand(w_sub: np.ndarray, cols: pd.Index) -> np.ndarray:
        w = np.zeros(md.n_assets)
        w[[md.assets.index(c) for c in cols]] = w_sub
        return np.append(w, 0.0)

    def target(k: int, w: np.ndarray) -> np.ndarray | None:
        if k not in flags:
            return None
        if kind in ("momentum", "lowvol"):
            need = bcfg["momentum_lookback_days"] if kind == "momentum" else bcfg["lowvol_window_days"]
            hist = estimation_window(risky_daily, md.dates[k], need, need)
            if hist is None:
                raise ValueError(f"historique insuffisant ({kind}) au {md.dates[k].date()}")
            hist = eligible(hist, k)
            if kind == "momentum":
                ws = momentum_weights(hist, need, bcfg["momentum_skip_days"], bcfg["momentum_top_k"])
            else:
                ws = lowvol_weights(hist, bcfg["lowvol_top_k"])
            return expand(ws, hist.columns)
        hist = estimation_window(risky_daily, md.dates[k], bcfg["cov_window_days"], bcfg["min_window_days"])
        if hist is None:
            raise ValueError(f"historique insuffisant pour estimer la covariance au {md.dates[k].date()}")
        hist = eligible(hist, k)
        cov = ledoit_wolf_cov(hist)
        if kind == "minvar":
            wr = min_variance_weights(cov, bcfg["solver"])
        elif kind == "riskparity":
            wr = risk_parity_weights(cov, bcfg["solver"])
        else:
            wr = max_sharpe_weights(hist.mean().to_numpy(), cov, rf_daily, bcfg["max_weight_sharpe"], bcfg["solver"])
        return expand(wr, hist.columns)

    return target


def _periodic(md: MarketData, split: str, cfg: dict[str, Any], kind: str) -> pd.DataFrame:
    """Stratégies rebalancées à `rebalance_frequency`, estimées sur une fenêtre glissante <= t."""
    return _sim(md, split, cfg, md.weekly_returns, periodic_target_fn(md, md.splits[split], cfg, kind), md.all_assets)


_BASE_CACHE: dict[tuple, dict[int, np.ndarray]] = {}


def base_weight_path(md: MarketData, idx: np.ndarray, cfg: dict[str, Any], kind: str) -> dict[int, np.ndarray]:
    """Poids cibles (titres seulement) de la stratégie `kind` pour chaque décision k de idx : recalculés aux
    dates de rebalancement, maintenus entre deux (portefeuille de référence de l'agent)."""
    # les poids ne dépendent que des prix et des dates (pas de la normalisation des features)
    key = (str(md.dates[idx[0]]), str(md.dates[idx[-1]]), len(idx), tuple(md.assets), kind,
           cfg["baselines"]["rebalance_frequency"], cfg["baselines"]["cov_window_days"])
    if key not in _BASE_CACHE:
        fn = periodic_target_fn(md, idx, cfg, kind)
        path, cur = {}, None
        for k in idx:
            t = fn(int(k), None)
            cur = t[:-1] if t is not None else cur
            path[int(k)] = cur
        _BASE_CACHE[key] = path
    return _BASE_CACHE[key]


def run_min_variance(md: MarketData, split: str, cfg: dict[str, Any]) -> pd.DataFrame:
    return _periodic(md, split, cfg, "minvar")


def run_max_sharpe(md: MarketData, split: str, cfg: dict[str, Any]) -> pd.DataFrame:
    return _periodic(md, split, cfg, "maxsharpe")


def run_risk_parity(md: MarketData, split: str, cfg: dict[str, Any]) -> pd.DataFrame:
    return _periodic(md, split, cfg, "riskparity")


def run_momentum(md: MarketData, split: str, cfg: dict[str, Any]) -> pd.DataFrame:
    return _periodic(md, split, cfg, "momentum")


def run_lowvol(md: MarketData, split: str, cfg: dict[str, Any]) -> pd.DataFrame:
    return _periodic(md, split, cfg, "lowvol")


PORTFOLIO_BASELINES: dict[str, Callable[[MarketData, str, dict[str, Any]], pd.DataFrame]] = {
    "EqualWeight_1N": run_equal_weight,
    "MinVariance": run_min_variance,
    "MaxSharpe": run_max_sharpe,
    "RiskParity": run_risk_parity,
    "Momentum_12_1": run_momentum,
    "LowVol": run_lowvol,
}


def run_all_baselines(md: MarketData, split: str, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Buy & hold de chaque benchmark puis les stratégies de portefeuille."""
    out = {}
    for j, ticker in enumerate(md.benchmark_names):
        name = f"BuyHold_{benchmark_label(ticker)}"
        logger.info("Baseline %s sur %s", name, split)
        out[name] = run_buy_and_hold(md, split, cfg, j)
    for name, fn in PORTFOLIO_BASELINES.items():
        if name not in cfg["baselines"]["strategies"]:
            continue
        logger.info("Baseline %s sur %s", name, split)
        out[name] = fn(md, split, cfg)
    return out
