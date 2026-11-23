"""Génère les figures du rapport (report/figures/*.pdf) à partir des résultats du projet.

Usage (depuis la racine du projet) : python report/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import drawdown_series  # noqa: E402
from src.plots import GRID, INK, INK2, OTHER, SERIES, SURFACE  # noqa: E402

TABLES, OUT = ROOT / "results" / "tables", ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "DejaVu Sans"})

LABELS = {
    "RL_ensemble": "Agent PPO", "Action nulle (non entraîné)": "Référence non entraînée",
    "EqualWeight_1N": "1/N", "MaxSharpe": "Max-Sharpe", "MinVariance": "Variance min.",
    "LowVol": "Faible vol.", "Momentum_12_1": "Momentum 12-1", "RiskParity": "Parité de risque",
    "BuyHold_EXW1": "ETF Euro Stoxx 50", "BuyHold_STOXX50E": "Indice (prix)",
}
ORDER = ["RL_ensemble", "EqualWeight_1N", "MaxSharpe", "MinVariance", "LowVol", "Momentum_12_1", "RiskParity", "BuyHold_EXW1"]
COLOR = {k: SERIES[i] for i, k in enumerate(ORDER)}


def style(ax: plt.Axes, title: str = "", ylabel: str = "", grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def save(fig: plt.Figure, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", facecolor=SURFACE)
    fig.savefig(OUT / f"{name}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def backtest(name: str) -> pd.DataFrame:
    df = pd.read_csv(TABLES / f"test_backtest_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                     index_col=0, parse_dates=["date", "end_date"])
    return df


# ------------------------------------------------------------------ 1. univers et découpage temporel
prices = pd.read_csv(ROOT / "data" / "processed" / "prices.csv", index_col=0, parse_dates=True)
raw = pd.read_csv(ROOT / "data" / "raw" / "close.csv", index_col=0, parse_dates=True).drop(columns=["^STOXX50E", "EXW1.DE"], errors="ignore")
raw = raw.reindex(prices.index).ffill(limit=3)
investable = (raw.notna().rolling(253).sum() >= 253).sum(axis=1)  # titres ayant >= 1 an de cotation
fig, ax = plt.subplots(figsize=(7.5, 2.6))
spans = [("2010-01-01", "2013-12-31", "Entraînement\n(1er bloc)", "#e9eef7"), ("2014-01-01", "2021-12-31", "Validation glissante\n(4 blocs de 2 ans)", "#fdf0e6"),
         ("2022-01-01", "2026-11-20", "Test unique", "#e7f5ef")]
for a, b, lab, col in spans:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=col, zorder=0)
    ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, 51.2, lab, ha="center", va="bottom", fontsize=8, color=INK2)
ax.plot(investable.index, investable.values, color=SERIES[0], linewidth=2)
ax.set_ylim(40, 56)
style(ax, "Titres investissables (au moins 1 an de cotation) et découpage temporel", "Nombre de titres")
save(fig, "fig_univers")

# ------------------------------------------------------------------ 2. valeur cumulée et drawdown au test
names = ["RL_ensemble", "EqualWeight_1N", "MaxSharpe", "MinVariance", "Momentum_12_1", "BuyHold_EXW1"]
bt = {n: backtest(n) for n in names}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
for n, df in bt.items():
    v = pd.concat([pd.Series([1.0], index=[df.index[0]]), (1 + df["net_return"]).cumprod().set_axis(df["end_date"])])
    ax1.plot(v.index, v.values, color=COLOR[n], linewidth=2.2 if n == "RL_ensemble" else 1.4, label=LABELS[n])
    dd = pd.Series(drawdown_series(df["net_return"].to_numpy()), index=df["end_date"])
    ax2.plot(dd.index, dd.values, color=COLOR[n], linewidth=2.2 if n == "RL_ensemble" else 1.2)
style(ax1, "Valeur cumulée nette de frais (base 1) — test 2022-2026", "Valeur")
ax1.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
style(ax2, "Drawdown (baisse depuis le dernier sommet)", "Drawdown")
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
save(fig, "fig_test_valeur_drawdown")

# ------------------------------------------------------------------ 3. intervalles de confiance du Sharpe (test)
tab = pd.read_csv(TABLES / "test_comparison.csv", index_col=0)
rows = [n for n in ORDER + ["BuyHold_STOXX50E"] if n in tab.index]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for i, n in enumerate(rows[::-1]):
    s, lo, hi = tab.loc[n, ["sharpe", "sharpe_ci_low", "sharpe_ci_high"]]
    c = COLOR.get(n, OTHER)
    ax.plot([lo, hi], [i, i], color=c, linewidth=2)
    ax.plot(s, i, "o", color=c, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.text(hi + 0.05, i, f"{s:.2f}", va="center", fontsize=8, color=INK)
ax.axvline(tab.loc["EqualWeight_1N", "sharpe"], color=INK2, linestyle="--", linewidth=1)
ax.set_yticks(range(len(rows)), [LABELS[n] for n in rows[::-1]])
style(ax, "Sharpe au test et intervalle de confiance à 95 % (bootstrap par blocs)", "", grid_axis="x")
ax.set_xlabel("Sharpe annualisé", color=INK2)
save(fig, "fig_test_ic_sharpe")

# ------------------------------------------------------------------ 4. rendement / risque (test)
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for n in rows:
    vol, ret = tab.loc[n, ["ann_vol", "ann_return"]]
    ax.scatter(vol, ret, s=70, color=COLOR.get(n, OTHER), edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.annotate(LABELS[n], (vol, ret), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
for sh in (0.5, 1.0, 1.5):
    x = np.linspace(0.10, 0.22, 10)
    ax.plot(x, sh * x, color=GRID, linewidth=1, zorder=1)
    ax.text(0.218, sh * 0.218, f"Sharpe {sh}", fontsize=7, color=INK2, ha="right", va="bottom")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
style(ax, "Rendement et volatilité annualisés — test 2022-2026", "Rendement annualisé", grid_axis="both")
ax.set_xlabel("Volatilité annualisée", color=INK2)
save(fig, "fig_test_rendement_risque")

# ------------------------------------------------------------------ 5. Sharpe par bloc de validation
folds = pd.read_csv(TABLES / "tuning_folds.csv")
strat = {"autour_maxsharpe · RL_ensemble": "RL_ensemble", "MaxSharpe": "MaxSharpe", "MinVariance": "MinVariance",
         "EqualWeight_1N": "EqualWeight_1N", "BuyHold_EXW1": "BuyHold_EXW1"}
blocks = ["2014-15", "2016-17", "2018-19", "2020-21"]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
width = 0.16
for j, (key, n) in enumerate(strat.items()):
    vals = folds[folds.strategy == key].sort_values("fold").sharpe.to_numpy()
    ax.bar(np.arange(4) + (j - 2) * width, vals, width * 0.92, color=COLOR[n], label=LABELS[n])
ax.set_xticks(range(4), blocks)
ax.axhline(0, color=GRID)
style(ax, "Sharpe par bloc de validation glissante", "Sharpe")
ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
save(fig, "fig_validation_blocs")

# ------------------------------------------------------------------ 6. réglage : 8 configurations
summ = pd.read_csv(TABLES / "tuning_summary.csv", index_col=0)
cfgs = ["base", "tilt_fort", "non_myope", "sans_cash", "rendement_pur", "autour_minvar", "autour_maxsharpe"]
lab = {"base": "autour de 1/N", "tilt_fort": "1/N, tilt fort", "non_myope": "1/N, γ=0,9", "sans_cash": "1/N, sans cash",
       "rendement_pur": "1/N, rendement pur", "autour_minvar": "autour min-var.", "autour_maxsharpe": "autour max-Sharpe"}
fig, ax = plt.subplots(figsize=(7.5, 3.2))
x = np.arange(len(cfgs))
tr = [summ.loc[f"{c} · RL_ensemble", "sharpe"] for c in cfgs]
nu = [summ.loc[f"{c} · Action nulle (non entraîné)", "sharpe"] for c in cfgs]
ax.bar(x - 0.2, nu, 0.38, color=OTHER, label="Référence non entraînée")
ax.bar(x + 0.2, tr, 0.38, color=SERIES[0], label="Agent entraîné")
ax.set_xticks(x, [lab[c] for c in cfgs], rotation=20, ha="right")
style(ax, "Réglage : Sharpe de validation (2014-2021) par configuration hebdomadaire", "Sharpe")
ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "fig_reglage")

# ------------------------------------------------------------------ 7. poids et cash de l'agent au test
rl = backtest("RL_ensemble")
wcols = [c for c in rl.columns if c.startswith("w_") and c != "w_CASH"]
avg = rl[wcols].mean().sort_values(ascending=False)
top = list(avg.index[:7])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.3), gridspec_kw={"width_ratios": [3, 2]})
data = rl[top].copy()
data["Autres"] = rl[[c for c in wcols if c not in top]].sum(axis=1) + rl["w_CASH"]
ax1.stackplot(rl.index, data.T.values, labels=[c[2:] for c in top] + ["Autres + cash"],
              colors=SERIES[:7] + [OTHER], edgecolor=SURFACE, linewidth=0.4)
ax1.set_ylim(0, 1)
ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
style(ax1, "Poids de l'agent (7 premiers titres)", "Poids")
ax1.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
avg_top = avg.iloc[:12][::-1]
ax2.barh([c[2:] for c in avg_top.index], avg_top.values, color=SERIES[0])
ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
style(ax2, "Poids moyen au test", "", grid_axis="x")
save(fig, "fig_poids_agent")

# ------------------------------------------------------------------ 8. turnover hebdomadaire
fig, ax = plt.subplots(figsize=(7.5, 2.8))
for n in ["RL_ensemble", "MaxSharpe", "EqualWeight_1N"]:
    df = backtest(n)
    t = df["turnover"].iloc[1:]
    ax.hist(t, bins=40, range=(0, 0.6), alpha=0.75, color=COLOR[n], label=f"{LABELS[n]} (moy. {t.mean():.1%})")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
style(ax, "Distribution du turnover hebdomadaire — test", "Semaines")
ax.set_xlabel("Part du portefeuille échangée dans la semaine", color=INK2)
ax.legend(frameon=False, fontsize=8)
save(fig, "fig_turnover")

# ------------------------------------------------------------------ 9. dynamique d'entraînement (5 seeds finales)
fig, ax = plt.subplots(figsize=(7.5, 2.8))
for s in range(5):
    h = pd.read_csv(ROOT / "results" / "models" / "final" / "autour_maxsharpe" / f"seed_{s}" / "eval_history.csv")
    ax.plot(h["timesteps"], h["val_sharpe"], color=SERIES[s], linewidth=1.6, marker="o", markersize=3, label=f"seed {s}")
style(ax, "Sharpe de diagnostic (2020-2021) pendant l'entraînement des modèles finaux", "Sharpe")
ax.set_xlabel("Pas d'entraînement", color=INK2)
ax.legend(frameon=False, fontsize=8, ncol=5)
save(fig, "fig_entrainement")

# ------------------------------------------------------------------ 10. versions explorées
versions = ["Agent sur 20 titres,\nrendements bruts", "Agent qui ne choisit\nque % actions / cash", "Même réseau appris\npar gradient direct",
            "Agent qui choisit\nentre 6 stratégies", "Modèle retenu :\n50 titres, autour du\nmax-Sharpe"]
test_sh = [0.58, 0.73, 0.93, 1.18, 1.05]
cols = [OTHER, OTHER, OTHER, OTHER, SERIES[0]]
fig, ax = plt.subplots(figsize=(7.5, 3.1))
ax.bar(range(5), test_sh, color=cols, width=0.6)
for i, v in enumerate(test_sh):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK)
ax.axhline(0.65, color=SERIES[7], linestyle="--", linewidth=1)
ax.text(4.45, 0.67, "ETF", color=INK2, fontsize=8, ha="right")
ax.set_xticks(range(5), versions, fontsize=7)
style(ax, "Sharpe au test des alternatives étudiées (le choix a été fait sur la validation)", "Sharpe")
save(fig, "fig_versions")

# ------------------------------------------------------------------ 11. d'où vient l'écart avec l'ETF
steps = [("ETF Euro Stoxx 50\n(pondéré par la\ncapitalisation)", "BuyHold_EXW1"), ("1/N : 50 titres\nà poids égaux", "EqualWeight_1N"),
         ("Max-Sharpe\n(référence)", "MaxSharpe"), ("Agent PPO", "RL_ensemble")]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0))
for ax, col, title, pct in ((ax1, "ann_return", "Rendement annualisé (test)", True), (ax2, "sharpe", "Sharpe (test)", False)):
    vals = [tab.loc[k, col] for _, k in steps]
    ax.bar(range(4), vals, color=[COLOR[k] for _, k in steps], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{100 * v:.1f} %" if pct else f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(4), [s for s, _ in steps], fontsize=6.5)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style(ax, title)
save(fig, "fig_ecart_etf")

# ------------------------------------------------------------------ 12. régularité : Sharpe sur 5 périodes
periods = ["2014-15", "2016-17", "2018-19", "2020-21", "Test\n2022-26"]
reg = {"RL_ensemble": "autour_maxsharpe · RL_ensemble", "MaxSharpe": "MaxSharpe", "MinVariance": "MinVariance",
       "LowVol": "LowVol", "EqualWeight_1N": "EqualWeight_1N", "BuyHold_EXW1": "BuyHold_EXW1"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.3), gridspec_kw={"width_ratios": [3, 2]})
for n, key in reg.items():
    v = list(folds[folds.strategy == key].sort_values("fold").sharpe) + [tab.loc[n, "sharpe"]]
    ax1.plot(range(5), v, color=COLOR[n], linewidth=2.4 if n == "RL_ensemble" else 1.3, marker="o", markersize=5,
             markeredgecolor=SURFACE, label=LABELS[n])
ax1.set_xticks(range(5), periods, fontsize=7.5)
style(ax1, "Sharpe par période", "Sharpe")
ax1.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
worst = {n: min(list(folds[folds.strategy == k].sharpe) + [tab.loc[n, "sharpe"]]) for n, k in reg.items()}
order = sorted(worst, key=worst.get)
ax2.barh([LABELS[n] for n in order], [worst[n] for n in order], color=[COLOR[n] for n in order])
for i, n in enumerate(order):
    ax2.text(worst[n] + 0.02, i, f"{worst[n]:.2f}", va="center", fontsize=8, color=INK)
style(ax2, "Sharpe de la pire période", "", grid_axis="x")
save(fig, "fig_regularite")

print(f"Figures écrites dans {OUT}")
