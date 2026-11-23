"""Phase 7 : tableaux comparatifs (CSV / Markdown) ; les figures sont dans src/plots.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import metrics_from_frame, uncertainty_columns

PCT_COLS = {"ann_return", "ann_vol", "max_drawdown", "avg_turnover", "cum_return", "avg_exposure"}
COL_LABELS = {
    "ann_return": "Rdt annualisé",
    "ann_vol": "Vol annualisée",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Max DD",
    "calmar": "Calmar",
    "avg_turnover": "Turnover moyen/sem.",
    "cum_return": "Rdt cumulé",
    "n_periods": "Semaines",
}


UNCERTAINTY_COLS = ["sharpe_ci_low", "sharpe_ci_high", "p_vs_ref"]


def metrics_table(results: dict[str, pd.DataFrame], cfg: dict[str, Any], uncertainty: bool = True) -> pd.DataFrame:
    """Une ligne de métriques par stratégie ; + IC bootstrap du Sharpe et p-value contre la référence."""
    table = pd.DataFrame({name: metrics_from_frame(df, cfg) for name, df in results.items()}).T
    if uncertainty:
        table = table.join(uncertainty_columns({n: df["net_return"].to_numpy() for n, df in results.items()}, cfg))
    return table


def _fmt(col: str, v: Any) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if col == "n_periods":
        return f"{int(v)}"
    return f"{100 * v:.2f} %" if col in PCT_COLS else f"{v:.2f}"


def to_markdown(table: pd.DataFrame, std: pd.DataFrame | None = None) -> str:
    """Tableau Markdown ; si `std` est fourni, affiche « moyenne ± écart-type » pour les lignes concernées."""
    cols = list(table.columns)
    def label(c: str) -> str:
        if c.startswith("sharpe_fold"):
            return f"Sharpe bloc {c[len('sharpe_fold'):]}"
        return {"sharpe_std_folds": "É.-t. Sharpe (blocs)", "sharpe_ci_low": "Sharpe IC95 bas",
                "sharpe_ci_high": "Sharpe IC95 haut", "p_vs_ref": "p (Sharpe ≠ réf.)", "avg_exposure": "Expo. moyenne",
                "gain_vs_1N": "Gain Sharpe vs 1/N"}.get(c, COL_LABELS.get(c, c))

    lines = ["| Stratégie | " + " | ".join(label(c) for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, row in table.iterrows():
        cells = []
        for c in cols:
            s = _fmt(c, row[c])
            if std is not None and name in std.index and c != "n_periods" and not pd.isna(std.loc[name, c]):
                s += " ± " + _fmt(c, std.loc[name, c])
            cells.append(s)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_table(table: pd.DataFrame, path_stem: Path, title: str, std: pd.DataFrame | None = None,
               notes: str = "") -> None:
    """Écrit <stem>.csv et <stem>.md."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    out = table.copy()
    if std is not None:
        out = out.join(std.add_suffix("_std"), how="left")
    out.to_csv(path_stem.with_suffix(".csv"), float_format="%.6f")
    md = f"# {title}\n\n" + to_markdown(table, std) + (f"\n{notes}\n" if notes else "")
    path_stem.with_suffix(".md").write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- verrou du test final
def file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_final_test_lock(lock: Path, force: bool) -> None:
    """Refuse un 2e passage sur le test sans --force."""
    if lock.exists() and not force:
        raise SystemExit(
            f"REFUS : le test final a déjà été exécuté ({lock.read_text(encoding='utf-8').strip()}).\n"
            "Le jeu de test ne doit être évalué qu'une fois. Utiliser --force uniquement en connaissance de cause."
        )
