"""Données : constituants (Wikipédia) -> tickers Yahoo vérifiés -> prix nettoyés + benchmarks + contexte de marché."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (  # noqa: E402
    add_cash,
    build_candidates,
    clean_prices,
    compute_returns,
    download_ohlcv,
    processed_paths,
    select_universe,
    verify_and_resolve,
    write_report,
)
from src.universe import download_context  # noqa: E402
from src.utils import ensure_dirs, load_config, project_path, set_global_seed, setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--refresh", action="store_true", help="re-télécharge même si le cache brut existe")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging(cfg, "01_download_data")
    set_global_seed(cfg["seed"])
    dcfg = cfg["data"]
    raw = project_path(cfg["paths"]["raw_dir"])
    f_tickers, f_close, f_vol = raw / "tickers_resolved.csv", raw / "close.csv", raw / "volume.csv"
    benchmarks = [dcfg["benchmark"], dcfg["benchmark_total_return"]]

    if args.refresh or not (f_tickers.exists() and f_close.exists() and f_vol.exists()):
        cands, source = build_candidates(cfg)
        log.info("%d sociétés candidates (source : %s)", len(cands), source)
        resolved = verify_and_resolve(cands, cfg)
        resolved["source"] = source
        resolved.to_csv(f_tickers, index=False)
        tickers = resolved["yahoo_ticker"].dropna().tolist()
        log.info("%d tickers valides ; téléchargement %s -> %s", len(tickers), dcfg["start_date"], dcfg["end_date"] or "aujourd'hui")
        close, volume = download_ohlcv(tickers + benchmarks, dcfg["start_date"], dcfg["end_date"], cfg)
        missing = [t for t in tickers + benchmarks if t not in close.columns]
        if any(b in missing for b in benchmarks):
            raise RuntimeError(f"Benchmark introuvable sur Yahoo Finance : {[b for b in benchmarks if b in missing]}")
        if missing:
            log.warning("Historique complet indisponible pour %s", missing)
        close.to_csv(f_close)
        volume.to_csv(f_vol)
    else:
        log.info("Cache brut trouvé dans %s (utiliser --refresh pour re-télécharger)", raw)

    resolved = pd.read_csv(f_tickers, dtype={"candidates": str, "failed": str})
    resolved[["candidates", "failed"]] = resolved[["candidates", "failed"]].fillna("")
    source = str(resolved["source"].iloc[0])
    close = pd.read_csv(f_close, index_col=0, parse_dates=True)
    volume = pd.read_csv(f_vol, index_col=0, parse_dates=True)
    new = [b for b in benchmarks if b not in close.columns]
    if new:  # benchmark ajouté après coup : téléchargé sur la MÊME période que le cache (pas de nouvelles dates)
        end = str((close.index.max() + pd.Timedelta(days=1)).date())
        log.info("Ajout au cache de %s (%s -> %s)", new, dcfg["start_date"], end)
        c_new, v_new = download_ohlcv(new, dcfg["start_date"], end, cfg)
        if any(b not in c_new.columns for b in new):
            raise RuntimeError(f"Benchmark introuvable sur Yahoo Finance : {new}")
        close = close.join(c_new[new], how="left")
        volume = volume.join(v_new[new], how="left")
        close.to_csv(f_close)
        volume.to_csv(f_vol)
    bench = close[benchmarks].copy()
    close = close.drop(columns=benchmarks)
    volume = volume.drop(columns=benchmarks, errors="ignore")
    log.info("Données brutes : %s (%s -> %s)", close.shape, close.index.min().date(), close.index.max().date())

    res = clean_prices(close, volume, bench, dcfg["start_date"], dcfg["coverage_tolerance_days"],
                       dcfg["max_missing_frac"], dcfg["max_ffill_days"])
    for t, why in res.removed.items():
        log.info("%s hors du calendrier de référence (%s) : intégré au modèle un an après sa 1re cotation", t, why)
    log.info("Après nettoyage : %d titres, %d jours (%d jours supprimés)", res.close.shape[1], len(res.close), res.dropped_days)

    train_start, train_end = cfg["split"]["train"]
    universe = select_universe(res.close, res.volume, dcfg["n_assets"], train_start, train_end)
    log.info("Univers (top %d valeur échangée, train uniquement) : %s", dcfg["n_assets"], list(universe.index))

    prices = add_cash(res.close[universe.index.tolist()], dcfg["cash_daily_return"], dcfg["cash_name"])
    returns = compute_returns(prices)
    benchmark = res.benchmark[benchmarks]

    out = processed_paths(cfg)
    prices.to_csv(out["prices"])
    returns.to_csv(out["returns"])
    benchmark.to_csv(out["benchmark"])
    names = dict(zip(resolved["yahoo_ticker"], resolved["name"]))
    pd.DataFrame({"ticker": universe.index, "name": [names.get(t, "") for t in universe.index],
                  "avg_value_traded_train": universe.values}).to_csv(out["universe"], index=False)
    end_eff = str(close.index.max().date()) if dcfg["end_date"] is None else dcfg["end_date"]
    write_report(out["report"], cfg, source, resolved, res, universe, prices, returns, end_eff)

    ctx = download_context(cfg, refresh=args.refresh)
    log.info("Contexte de marché (indice + VIX) : %s", ctx)

    # Point de contrôle
    n_nan = int(prices.isna().sum().sum() + returns.isna().sum().sum() + benchmark.isna().sum().sum())
    print("\n=== Point de contrôle Phase 1 ===")
    print(f"prices    : {prices.shape}  {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"returns   : {returns.shape}")
    print(f"benchmark : {benchmark.shape}")
    print(f"NaN total : {n_nan}")
    print(f"Actifs    : {list(prices.columns)}")
    print(f"Rapport   : {out['report']}  (généré le {date.today()})")
    if n_nan:
        raise SystemExit("ERREUR : des NaN subsistent dans les données traitées")


if __name__ == "__main__":
    main()
