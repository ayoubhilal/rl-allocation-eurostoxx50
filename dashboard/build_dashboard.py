"""Construit l'interface interactive (dashboard/index.html) à partir des modèles finaux et des résultats.

Rejoue l'ensemble des 5 agents sur toute la période de test (2022 -> fin des données) pour récupérer, semaine par
semaine, les poids exécutés, les poids cibles, le portefeuille de référence et les rendements de chaque titre,
puis calcule l'allocation de la semaine en cours (décidée à la dernière date disponible, rendement encore inconnu).
Les rendements obtenus sont comparés à results/tables/test_backtest_RL_ensemble.csv (doivent être identiques).

Usage (depuis la racine du projet) :
    python dashboard/build_dashboard.py              # rejoue les modèles et reconstruit la page
    python dashboard/build_dashboard.py --html-only  # reconstruit la page à partir de dashboard/data.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import base_weight_path, run_all_baselines  # noqa: E402
from src.env import PortfolioEnv  # noqa: E402
from src.train import get_factory  # noqa: E402
from src.utils import load_config, project_path  # noqa: E402

_spec = importlib.util.spec_from_file_location("runner", ROOT / "scripts" / "02_train_evaluate.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

BENCHMARKS = {  # fichier de backtest -> libellé affiché
    "BuyHold_EXW1": "ETF Euro Stoxx 50",
    "EqualWeight_1N": "1/N (50 titres)",
    "MaxSharpe": "Max-Sharpe",
    "MinVariance": "Variance minimale",
    "LowVol": "Faible volatilité",
    "Momentum_12_1": "Momentum 12-1",
    "RiskParity": "Parité de risque",
    "BuyHold_STOXX50E": "Indice (prix)",
}


def r5(x: np.ndarray) -> list[float]:
    return [round(float(v), 5) for v in x]


def write_html(data: dict) -> Path:
    """Injecte les données dans le gabarit et écrit dashboard/index.html (fichier autonome)."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    template = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    out = ROOT / "dashboard" / "index.html"
    out.write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")
    return out


def main() -> None:
    if "--html-only" in sys.argv:  # reconstruit la page à partir de dashboard/data.json, sans recharger les modèles
        out = write_html(json.loads((ROOT / "dashboard" / "data.json").read_text(encoding="utf-8")))
        print(f"Interface écrite : {out} ({out.stat().st_size // 1024} Ko)")
        return
    cfg = load_config()
    tables = project_path(cfg["paths"]["tables_dir"])
    best = json.loads((tables / "best_config.json").read_text(encoding="utf-8"))
    cb = runner.with_overrides(cfg, best["overrides"])
    cb["split"]["train"] = list(cfg["walk_forward"]["final_train"])
    md = runner.build_md(cb)

    factory = get_factory(cb)
    root = project_path(cfg["paths"]["models_dir"]) / "final" / best["name"]
    models = [factory.load(root / f"seed_{s}" / "final_model.zip") for s in cfg["rl"]["seeds"]]
    acts = [(lambda m: (lambda o: m.predict(o, deterministic=True)[0]))(m) for m in models]

    # environnement de test prolongé d'une semaine : la dernière date disponible (rendement inconnu)
    env = PortfolioEnv(md, "test", cb, "eval")
    test_idx = md.splits["test"]
    ext = np.append(test_idx, len(md.dates) - 1)
    env.idx = ext
    env._base = base_weight_path(md, ext, cb, cb["env"]["base_portfolio"])
    obs, _ = env.reset(seed=0)

    weeks = []
    for _ in range(len(test_idx)):
        k = env.current_k
        target = np.mean([env.action_to_weights(a(obs)) for a in acts], axis=0)
        obs, _, _, _, info = env.step_target(target)
        weeks.append({
            "date": str(md.dates[k].date()), "end": str(md.dates[k + 1].date()),
            "w": r5(info["weights"]), "target": r5(target), "base": r5(env._base[k]),
            "avail": [int(v) for v in md.available[k]],
            "ret": round(float(info["net_return"]), 6), "gross": round(float(info["gross_return"]), 6),
            "turnover": round(float(info["turnover"]), 5), "R": r5(md.weekly_returns[k, :-1]),
        })
    # semaine en cours : décision à la dernière date, rendement pas encore connu
    k = env.current_k
    target = np.mean([env.action_to_weights(a(obs)) for a in acts], axis=0)
    alpha = float(cb["env"]["smoothing_alpha"])
    w_live = (1 - alpha) * env._weights + alpha * target
    turnover_live = float(np.abs(w_live - env._weights)[:-1].sum())
    weeks.append({"date": str(md.dates[k].date()), "end": None, "w": r5(w_live), "target": r5(target),
                  "base": r5(env._base[k]), "avail": [int(v) for v in md.available[k]], "ret": None,
                  "gross": None, "turnover": round(turnover_live, 5), "R": None})

    # contrôle : identique au backtest officiel sur les semaines communes (les données ont pu être prolongées)
    official = pd.read_csv(tables / "test_backtest_RL_ensemble.csv", index_col=0)["net_return"].to_numpy()
    mine = np.array([w["ret"] for w in weeks[:-1]])
    common = min(len(mine), len(official))
    gap = float(np.max(np.abs(mine[:common] - official[:common])))
    if gap > 1e-6:
        raise SystemExit(f"Écart avec le backtest officiel : {gap}")

    # stratégies de référence recalculées sur les données courantes (mêmes dates que l'agent)
    base = run_all_baselines(md, "test", cb)
    bench = {label: [round(float(v), 6) for v in base[key]["net_return"].to_numpy()] + [None]
             for key, label in BENCHMARKS.items()}

    names = pd.read_csv(project_path(cfg["paths"]["raw_dir"]) / "tickers_resolved.csv").set_index("yahoo_ticker")["name"]
    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": "Agent PPO - ensemble de 5 modèles, inclinaisons autour du max-Sharpe",
        "tickers": md.assets,
        "names": {t: str(names.get(t, t)) for t in md.assets},
        "cost": float(cb["env"]["transaction_cost"]),
        "weeks": weeks,
        "bench": bench,
    }
    (ROOT / "dashboard" / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    out = write_html(data)
    print(f"{len(weeks)} semaines ({weeks[0]['date']} -> {weeks[-1]['date']}), écart au backtest officiel {gap:.1e}")
    print(f"Interface écrite : {out} ({out.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    main()
