"""Calendrier de décision hebdomadaire, découpage temporel, structure MarketData, normalisation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("train", "validation", "test")


def weekly_decision_dates(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    """Dernier jour de bourse de chaque semaine civile."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period(rule)).max().values)


def weekly_asset_returns(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Rendement de chaque actif entre deux dates de décision consécutives : ligne k = (t_k, t_{k+1}]."""
    p = prices.loc[dates]
    return (p.shift(-1) / p - 1.0).iloc[:-1]


def split_transitions(dates: pd.DatetimeIndex, start: str, end: str | None) -> np.ndarray:
    """Indices k des transitions (décision t_k, période de détention (t_k, t_{k+1}]) incluses dans [start, end]."""
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end is not None else dates.max()
    k = np.arange(len(dates) - 1)
    mask = (dates[:-1] >= lo) & (dates[1:] <= hi)
    return k[np.asarray(mask)]


@dataclass
class Normalizer:
    """Z-score par groupe de features ; statistiques par actif (per_asset) ou communes à tous (pooled)."""

    mean: np.ndarray  # (N, F)
    std: np.ndarray   # (N, F)
    clip: float

    @classmethod
    def fit(cls, x: np.ndarray, n_pooled: int, clip: float, pool_assets: bool = False) -> "Normalizer":
        """x : (T, N, F) features brutes des dates du train uniquement.

        Les `n_pooled` premières features forment un seul groupe (lags de rendement), les autres un groupe chacune.
        """
        _, n, f = x.shape
        mean, std = np.zeros((n, f)), np.ones((n, f))
        groups = ([slice(0, n_pooled)] if n_pooled > 0 else []) + [slice(j, j + 1) for j in range(n_pooled, f)]
        for g in groups:
            block = x[:, :, g]  # (T, N, g)
            axes = (0, 1, 2) if pool_assets else (0, 2)
            m = np.broadcast_to(block.mean(axis=axes), (n,))
            s = np.broadcast_to(block.std(axis=axes), (n,))
            mean[:, g] = m[:, None]
            std[:, g] = np.where(s > 1e-12, s, 1.0)[:, None]
        return cls(mean=mean, std=std, clip=clip)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip)


@dataclass
class MarketData:
    """Tout ce dont l'environnement et les baselines ont besoin, aligné sur les dates de décision."""

    dates: pd.DatetimeIndex          # K dates de décision
    assets: list[str]                # N actifs risqués
    cash_name: str
    features: np.ndarray             # (K, N, F) normalisées
    raw_features: np.ndarray         # (K, N, F) brutes
    weekly_returns: np.ndarray       # (K-1, N+1) ; dernière colonne = CASH
    splits: dict[str, np.ndarray]    # indices de transitions par split
    daily_returns: pd.DataFrame      # rendements journaliers (N + CASH)
    benchmark_weekly: np.ndarray     # (K-1, B) rendement hebdo de chaque benchmark
    normalizer: Normalizer
    benchmark_names: list[str]
    available: np.ndarray | None = None  # (K, N) titres investissables (None = tous, univers fixe)

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def all_assets(self) -> list[str]:
        return self.assets + [self.cash_name]

