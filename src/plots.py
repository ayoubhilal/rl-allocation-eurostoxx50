"""Figures (matplotlib, PNG). Palette catégorielle validée, ordre fixe, au plus 8 séries."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.metrics import drawdown_series  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b4b3ad"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def _style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _legend(ax: plt.Axes, **kw) -> None:
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK2, **kw)
    return leg


def plot_lines(series: dict[str, pd.Series], path: Path, title: str, ylabel: str,
               bands: dict[str, tuple[pd.Series, pd.Series]] | None = None, pct: bool = False,
               markers: bool = False) -> None:
    """Courbes (une par stratégie), couleur fixée par l'ordre d'insertion ; bande optionnelle (min-max seeds)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, s) in enumerate(series.items()):
        color = SERIES[i % len(SERIES)]
        if bands and name in bands:
            lo, hi = bands[name]
            ax.fill_between(lo.index, lo.values, hi.values, color=color, alpha=0.15, linewidth=0)
        ax.plot(s.index, s.values, color=color, linewidth=2 if i == 0 else 1.6, label=name,
                marker="o" if markers else None, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax, title, ylabel)
    _legend(ax, loc="best")
    _save(fig, path)


def value_curves(results: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Valeur cumulée (base 1 à la 1re date de décision), indexée par date de fin de période."""
    out = {}
    for name, df in results.items():
        v = (1 + df["net_return"]).cumprod()
        v.index = pd.to_datetime(df["end_date"])
        start = pd.Series([1.0], index=[pd.to_datetime(df.index[0])])
        out[name] = pd.concat([start, v])
    return out


def drawdown_curves(results: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    out = {}
    for name, df in results.items():
        out[name] = pd.Series(drawdown_series(df["net_return"].to_numpy()), index=pd.to_datetime(df["end_date"]))
    return out


def plot_weights(weights: pd.DataFrame, path: Path, title: str, top: int) -> None:
    """Aire empilée des poids moyens ; au-delà de `top` actifs (+ CASH), regroupement en « Autres »."""
    top = min(top, len(SERIES) - 1)
    avg = weights.mean().sort_values(ascending=False)
    keep = list(avg.index[:top])
    rest = [c for c in weights.columns if c not in keep]
    cols = keep + (["Autres"] if rest else [])
    data = weights[keep].copy()
    if rest:
        data["Autres"] = weights[rest].sum(axis=1)
    colors = SERIES[: len(keep)] + ([OTHER] if rest else [])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(data.index, data.T.values, labels=[c.replace("w_", "") for c in cols], colors=colors,
                 edgecolor=SURFACE, linewidth=0.6)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax, title, "Poids")
    _legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, path)


def plot_turnover_hist(turnovers: dict[str, pd.Series], path: Path, title: str, bins: int) -> None:
    """Histogrammes du turnover hebdomadaire (petits multiples, un panneau par stratégie)."""
    n = len(turnovers)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    for i, (ax, (name, s)) in enumerate(zip(axes, turnovers.items())):
        edges = np.linspace(0, max(float(s.max()), 0.01), bins + 1)
        ax.hist(s.values, bins=edges, color=SERIES[i % len(SERIES)], edgecolor=SURFACE, linewidth=1)
        _style(ax, name, "Semaines" if i == 0 else "")
        ax.set_xlabel("Turnover hebdomadaire", color=INK2, fontsize=9)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.axvline(s.mean(), color=INK, linewidth=1, linestyle="--")
        ax.text(s.mean(), ax.get_ylim()[1] * 0.92, f" moy. {s.mean():.1%}", color=INK, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    _save(fig, path)
