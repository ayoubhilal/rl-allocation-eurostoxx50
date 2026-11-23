"""Modèle : réglage sur validation glissante, puis entraînement final et test unique.

  --tune        : pour chaque configuration de `tuning.grid`, entraîne `tuning.seeds` seeds sur chaque bloc de
                  `walk_forward.folds`, évalue l'ensemble sur la validation du bloc ; choisit la meilleure
                  configuration (Sharpe des blocs mis bout à bout) -> results/tables/best_config.json
  --final-test  : entraîne la meilleure configuration (toutes les seeds) sur walk_forward.final_train et l'évalue
                  UNE fois sur le test (verrou dédié ; --force pour outrepasser)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import run_all_baselines  # noqa: E402
from src.data import load_processed  # noqa: E402
from src.env import PortfolioEnv, ensemble_rollout, rollout  # noqa: E402
from src.evaluate import check_final_test_lock, metrics_table, save_table, to_markdown  # noqa: E402
from src.features import MarketData  # noqa: E402
from src.plots import drawdown_curves, plot_lines, plot_turnover_hist, plot_weights, value_curves  # noqa: E402
from src.train import get_factory, train_agent  # noqa: E402
from src.universe import build_full_universe, build_market_data  # noqa: E402
from src.utils import ensure_dirs, load_config, project_path, setup_logging  # noqa: E402

ENS, NULL = "RL_ensemble", "Action nulle (non entraîné)"


def with_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    for path, value in overrides.items():
        node = c
        *parents, leaf = path.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"clé inconnue dans la config : {path}")
        node[leaf] = value
    c["metrics"]["periods_per_year"] = 12 if c["features"]["weekly_rule"] == "M" else 52  # annualisation cohérente
    return c


def fold_cfg(cfg: dict[str, Any], fold: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    c["split"]["train"], c["split"]["validation"] = list(fold["train"]), list(fold["validation"])
    c["rl"]["total_timesteps"] = int(cfg["walk_forward"]["timesteps"])
    c["rl"]["checkpoint_selection"] = cfg["walk_forward"]["checkpoint_selection"]
    return c


def build_md(cfg: dict[str, Any]) -> MarketData:
    px = build_full_universe(cfg)
    _, _, bench = load_processed(cfg)
    return build_market_data(px, bench, cfg)


def _job(cfg_f: dict[str, Any], seed: int, out: str, tb: str, log_name: str) -> dict[str, Any]:
    setup_logging(cfg_f, log_name)
    if (Path(out) / "final_model.zip").exists():
        return {"run": tb, "seed": seed, "reused": True}
    return {"run": tb, **train_agent(cfg_f, build_md(cfg_f), seed, Path(out), tb)}


def run_jobs(jobs: list[tuple], n_jobs: int) -> list[dict]:
    if n_jobs <= 1:
        return [_job(*j) for j in jobs]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(_job, *zip(*jobs)))


def evaluate(cfg_f: dict[str, Any], md: MarketData, split: str, model_dirs: list[Path]) -> dict[str, pd.DataFrame]:
    """Ensemble, seeds, action nulle et baselines sur `split`."""
    factory = get_factory(cfg_f)
    models = [factory.load(d / "final_model.zip") for d in model_dirs]
    acts = [(lambda m: (lambda obs: m.predict(obs, deterministic=True)[0]))(m) for m in models]
    env = lambda: PortfolioEnv(md, split, cfg_f, "eval")  # noqa: E731
    res = {ENS: ensemble_rollout(env(), acts)}
    for d, a in zip(model_dirs, acts):
        res[f"RL {d.name}"] = rollout(env(), a)
    res[NULL] = rollout(env(), lambda obs: np.zeros(md.n_assets + 1, dtype=np.float32))
    res.update(run_all_baselines(md, split, cfg_f))
    return res


def ppy_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Copie de la config avec le bon nombre de périodes par an (52 hebdomadaire, 12 mensuel)."""
    c = copy.deepcopy(cfg)
    c["metrics"]["periods_per_year"] = 12 if cfg["features"]["weekly_rule"] == "M" else 52
    return c


def main_rows(res: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {k: v for k, v in res.items() if not k.startswith("RL seed")}


# --------------------------------------------------------------------------- réglage
def tune(cfg: dict[str, Any], log: Any) -> None:
    tables = project_path(cfg["paths"]["tables_dir"])
    seeds, folds, grid = cfg["tuning"]["seeds"], cfg["walk_forward"]["folds"], cfg["tuning"]["grid"]
    root = project_path(cfg["paths"]["models_dir"]) / "tuning"
    jobs = []
    for g in grid:
        cg = with_overrides(cfg, g["overrides"])
        for i, f in enumerate(folds):
            for s in seeds:
                jobs.append((fold_cfg(cg, f), s, str(root / g["name"] / f"fold_{i}" / f"seed_{s}"),
                             f"{g['name']}_fold{i}_seed{s}", f"02_tune_{g['name']}_fold{i}_seed{s}"))
    log.info("Réglage : %d configurations x %d blocs x %d seeds = %d entraînements", len(grid), len(folds), len(seeds), len(jobs))
    t0 = time.time()
    run_jobs(jobs, int(cfg["rl"]["n_jobs"]))
    setup_logging(cfg, "02_tune_eval")
    log.info("Entraînements terminés en %.1f min", (time.time() - t0) / 60)

    rows, pooled = [], {}
    for g in grid:
        cg = with_overrides(cfg, g["overrides"])
        for i, f in enumerate(folds):
            cf = fold_cfg(cg, f)
            md = build_md(cf)
            res = evaluate(cf, md, "validation", [root / g["name"] / f"fold_{i}" / f"seed_{s}" for s in seeds])
            for name, df in res.items():
                is_rl = name == ENS or name == NULL or name.startswith("RL ")
                if not is_rl and g is not grid[0]:
                    continue  # baselines : évaluées une fois (configuration de base, hebdomadaire)
                label = f"{g['name']} · {name}" if is_rl else name
                rows.append({"config": g["name"], "fold": i, "strategy": label,
                             "sharpe": metrics_table({label: df}, ppy_cfg(cf), False)["sharpe"].iloc[0]})
                if not name.startswith("RL "):
                    pooled.setdefault(label, []).append(df)
        log.info("Configuration %s évaluée", g["name"])
    pd.DataFrame(rows).to_csv(tables / "tuning_folds.csv", index=False)
    cat = {k: pd.concat(v) for k, v in pooled.items()}
    # chaque configuration mensuelle a son propre calendrier : comparaison par Sharpe annualisé
    # (52 périodes/an en hebdomadaire, 12 en mensuel)
    out = []
    for label, df in cat.items():
        g = next((x for x in grid if label.startswith(x["name"] + " · ")), None)
        cg = with_overrides(cfg, g["overrides"]) if g else cfg
        t = metrics_table({label: df}, ppy_cfg(cg), uncertainty=False)
        t["config"] = g["name"] if g else "baseline"
        out.append(t)
    summary = pd.concat(out)
    folds_tab = pd.DataFrame(rows).pivot_table(index="strategy", columns="fold", values="sharpe").add_prefix("sharpe_fold")
    summary = summary.join(folds_tab)
    # critère : gain de Sharpe sur 1/N MESURÉ À LA MÊME FRÉQUENCE (un Sharpe mensuel n'est pas comparable à un
    # Sharpe hebdomadaire) ; en mensuel, la référence est l'action nulle de la configuration (= 1/N mensuel)
    gains = {}
    for g in grid:
        cg = with_overrides(cfg, g["overrides"])
        if cg["features"]["weekly_rule"] == cfg["features"]["weekly_rule"]:
            ref = summary.loc["EqualWeight_1N", "sharpe"]
        else:
            if cg["env"]["base_portfolio"] != "equal":
                raise ValueError("configuration non hebdomadaire : la référence suppose base_portfolio = equal")
            ref = summary.loc[f"{g['name']} · {NULL}", "sharpe"]
        gains[g["name"]] = summary.loc[f"{g['name']} · {ENS}", "sharpe"] - ref
    summary["gain_vs_1N"] = [gains.get(n.split(" · ")[0], np.nan) if n.endswith(ENS) else np.nan for n in summary.index]
    best = max(gains, key=gains.get)
    summary = summary.sort_values("sharpe", ascending=False)
    notes = (f"Validation glissante {folds[0]['validation'][0]} -> {folds[-1]['validation'][1]} ({len(folds)} blocs), "
             f"{len(seeds)} seeds par bloc, {cfg['walk_forward']['timesteps']} pas, dernier checkpoint. "
             f"Critère : gain de Sharpe de l'ensemble sur 1/N à la même fréquence. "
             f"Meilleure configuration : **{best}** (gain {gains[best]:+.2f}). Attention : choisir le maximum de "
             f"{len(grid)} configurations surestime légèrement sa performance.")
    save_table(summary.drop(columns=["config"]), tables / "tuning_summary", "Réglage - validation glissante", notes=notes)
    (tables / "best_config.json").write_text(json.dumps(
        {"name": best, "overrides": next(g["overrides"] for g in grid if g["name"] == best),
         "selected_at": datetime.now().isoformat(), "gain_vs_1N": float(gains[best]),
         "criterion": "gain de Sharpe de l'ensemble sur 1/N (même fréquence), blocs mis bout à bout"},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Réglage (validation glissante, blocs mis bout à bout) ===\n")
    print(to_markdown(summary.drop(columns=["config"])))
    print(f"\nMeilleure configuration : {best}")


# --------------------------------------------------------------------------- test final
def final_test(cfg: dict[str, Any], force: bool, log: Any) -> None:
    lock = project_path(cfg["evaluation"]["final_test_lock"])
    check_final_test_lock(lock, force)
    tables, figs = project_path(cfg["paths"]["tables_dir"]), project_path(cfg["paths"]["figures_dir"])
    best = json.loads((tables / "best_config.json").read_text(encoding="utf-8"))
    cb = with_overrides(cfg, best["overrides"])
    cb["split"]["train"] = list(cfg["walk_forward"]["final_train"])
    cb["split"]["validation"] = list(cfg["walk_forward"]["folds"][-1]["validation"])  # diagnostic uniquement
    cb["rl"]["total_timesteps"] = int(cfg["walk_forward"]["timesteps"])
    cb["rl"]["checkpoint_selection"] = "final"
    root = project_path(cfg["paths"]["models_dir"]) / "final" / best["name"]
    seeds = list(cfg["rl"]["seeds"])
    run_jobs([(cb, s, str(root / f"seed_{s}"), f"final_{best['name']}_seed{s}", f"02_final_seed{s}") for s in seeds],
             int(cfg["rl"]["n_jobs"]))
    setup_logging(cfg, "02_final_test", mode="a")
    log.warning("ÉVALUATION FINALE SUR LE TEST (configuration %s)", best["name"])
    md = build_md(cb)
    res = evaluate(cb, md, "test", [root / f"seed_{s}" for s in seeds])
    tab = metrics_table(res, ppy_cfg(cb))
    cash = {k: float(v["w_CASH"].mean()) for k, v in res.items() if "w_CASH" in v}
    tab["avg_exposure"] = [1.0 - cash.get(n, 0.0) for n in tab.index]
    idx = md.splits["test"]
    notes = (f"Configuration « {best['name']} » choisie sur la validation glissante 2014-2021 ; modèles entraînés sur "
             f"{cfg['walk_forward']['final_train']} ; décisions {md.dates[idx[0]].date()} -> {md.dates[idx[-1]].date()}. "
             f"Le test 2022+ avait déjà été observé lors des étapes précédentes du projet : résultat indicatif. "
             f"Exécuté le {datetime.now():%Y-%m-%d %H:%M}{' (FORCÉ)' if force and lock.exists() else ''}.")
    save_table(tab, tables / "test_comparison", "TEST 2022+", notes=notes)
    for name, df in res.items():
        df.to_csv(tables / f"test_backtest_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv")
    main = main_rows(res)
    show = {k: main[k] for k in [ENS, NULL, "EqualWeight_1N", "BuyHold_EXW1", "Momentum_12_1", "MaxSharpe", "RiskParity", "LowVol"] if k in main}
    plot_lines(value_curves(show), figs / "test_value.png", "Valeur cumulée nette (test)", "Valeur (base 1)")
    plot_lines(drawdown_curves(show), figs / "test_drawdown.png", "Drawdowns (test)", "Drawdown", pct=True)
    wcols = [f"w_{a}" for a in md.all_assets]
    plot_weights(res[ENS][wcols], figs / "test_weights.png", "Poids de l'agent (test)", cfg["evaluation"]["top_assets_in_weight_plot"])
    plot_turnover_hist({k: v["turnover"].iloc[1:] for k, v in show.items()}, figs / "test_turnover.png",
                       "Turnover (test)", cfg["evaluation"]["turnover_bins"])
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"run_at": datetime.now().isoformat(), "config": best["name"], "forced": force}), encoding="utf-8")
    print(f"\n=== TEST 2022+ (configuration {best['name']}) ===\n")
    print(to_markdown(tab))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timesteps", type=int, default=None, help="surcharge walk_forward.timesteps (smoke test)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.timesteps is not None:
        cfg["walk_forward"]["timesteps"] = args.timesteps
    ensure_dirs(cfg)
    log = setup_logging(cfg, "02_final_test" if args.final_test else "02_tune", mode="a" if args.final_test else "w")
    t0 = time.time()
    if args.final_test:
        log.info("Tentative de test final (force=%s)", args.force)
        final_test(cfg, args.force, log)
    elif args.tune:
        tune(cfg, log)
    else:
        parser.error("préciser --tune ou --final-test")
    log.info("Durée : %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
