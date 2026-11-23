"""Univers complet (les 50 constituants) avec entrées progressives, signaux cross-sectionnels documentés
(momentum, retournement, faible volatilité, drawdown) et contexte de marché (tendance, volatilité, VIX).

Un titre n'est investissable à la date t que si TOUTES ses features sont calculables avec ses prix <= t
(il faut donc ~1 an de cotation). Les rangs sont calculés parmi les seuls titres disponibles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import MarketData, Normalizer, split_transitions, weekly_asset_returns, weekly_decision_dates
from src.utils import project_path

logger = logging.getLogger(__name__)

AVAIL = "available"


# --------------------------------------------------------------------------- données
def build_full_universe(cfg: dict[str, Any]) -> pd.DataFrame:
    """Prix ajustés des constituants (NaN avant la 1re cotation) sur le calendrier des données traitées."""
    raw = project_path(cfg["paths"]["raw_dir"])
    close = pd.read_csv(raw / "close.csv", index_col=0, parse_dates=True)
    close = close.drop(columns=[cfg["data"]["benchmark"], cfg["data"]["benchmark_total_return"]], errors="ignore")
    resolved = pd.read_csv(raw / "tickers_resolved.csv")
    tickers = [t for t in resolved["yahoo_ticker"].dropna() if t in close.columns]
    calendar = pd.read_csv(project_path(cfg["paths"]["processed_dir"]) / "prices.csv", index_col=0, parse_dates=True).index
    lim = cfg["data"]["max_ffill_days"]
    px = close[tickers].reindex(close.index.union(calendar)).ffill(limit=lim).reindex(calendar)
    for t in tickers:
        first = px[t].first_valid_index()
        if first is None or px[t].loc[first:].isna().any():
            raise ValueError(f"{t} : historique troué après sa 1re cotation")
    return px


def download_context(cfg: dict[str, Any], refresh: bool = False) -> Path:
    """Télécharge (ou réutilise) l'historique de l'indice et du VIX utilisé comme contexte de marché."""
    cc = cfg["features"]["context"]
    path = project_path(cfg["paths"]["raw_dir"]) / cc["cache_file"]
    if path.exists() and not refresh:
        return path
    import yfinance as yf

    tickers = [cc["index"], cc["vix"]]
    raw = yf.download(tickers, start=cc["download_start"], end=cc["download_end"], auto_adjust=True, progress=False)
    close = raw["Close"][tickers]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    if close.isna().all().any():
        raise RuntimeError(f"contexte de marché introuvable sur Yahoo Finance : {tickers}")
    path.parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(path)
    return path


def load_context(cfg: dict[str, Any], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Indice et VIX alignés sur le calendrier."""
    cc = cfg["features"]["context"]
    mkt = pd.read_csv(project_path(cfg["paths"]["raw_dir"]) / cc["cache_file"], index_col=0, parse_dates=True)
    lim = cfg["data"]["max_ffill_days"]
    return mkt[[cc["index"], cc["vix"]]].reindex(mkt.index.union(calendar)).ffill(limit=lim).reindex(calendar)


# --------------------------------------------------------------------------- features
def asset_feature_names(cfg: dict[str, Any]) -> list[str]:
    fc = cfg["features"]
    return [f"rank_{n}" for n in fc["rank_signals"]] + [f"ts_{n}" for n in fc["level_signals"]]


def context_feature_names(cfg: dict[str, Any]) -> list[str]:
    return ["mkt_trend", "mkt_vol", "mkt_ret_13w", "vix_level", "vix_chg"]


def feature_names(cfg: dict[str, Any]) -> list[str]:
    """Features par titre = signaux propres + contexte de marché (identique pour tous) + drapeau de disponibilité."""
    return asset_feature_names(cfg) + context_feature_names(cfg) + [AVAIL]


def masked_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rang dans [-1, 1] parmi les titres non-NaN de chaque date (NaN conservés)."""
    n = df.notna().sum(axis=1)
    r = df.rank(axis=1)
    return (r - 1.0).div((n - 1).where(n > 1), axis=0) * 2.0 - 1.0


def _signals(px: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Signaux bruts par titre (données <= t uniquement)."""
    fc = cfg["features"]
    s, lb = fc["signal_skip_days"], fc["signal_windows_days"]
    r = px.pct_change(fill_method=None)
    out = {
        "mom_12_1": px.shift(s) / px.shift(lb["mom_12"]) - 1.0,
        "mom_6_1": px.shift(s) / px.shift(lb["mom_6"]) - 1.0,
        "rev_1m": px / px.shift(s) - 1.0,
        "ret_1w": px / px.shift(5) - 1.0,
        "vol": r.rolling(lb["vol"]).std() * np.sqrt(fc["annualization_days"]),
        "drawdown": px / px.rolling(lb["drawdown"]).max() - 1.0,
    }
    unknown = set(fc["rank_signals"] + fc["level_signals"]) - set(out)
    if unknown:
        raise ValueError(f"signaux inconnus : {unknown}")
    return out


def compute_daily_features(px: pd.DataFrame, ctx: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """(features par titre {nom: DataFrame dates x titres}, contexte de marché DataFrame dates x 5)."""
    fc, cc = cfg["features"], cfg["features"]["context"]
    sig = _signals(px, cfg)
    feats = {f"rank_{n}": masked_rank(sig[n]) for n in fc["rank_signals"]}
    feats.update({f"ts_{n}": sig[n] for n in fc["level_signals"]})
    p, vix = ctx[cc["index"]], ctx[cc["vix"]]
    rm = p.pct_change(fill_method=None)
    context = pd.DataFrame({
        "mkt_trend": p / p.rolling(cc["ma_days"]).mean() - 1.0,
        "mkt_vol": rm.rolling(cc["vol_days"]).std() * np.sqrt(fc["annualization_days"]),
        "mkt_ret_13w": p / p.shift(65) - 1.0,
        "vix_level": vix,
        "vix_chg": np.log(vix / vix.shift(cc["vol_days"])),
    })
    return feats, context


def build_market_data(px: pd.DataFrame, bench: pd.DataFrame, cfg: dict[str, Any]) -> MarketData:
    """MarketData avec univers dynamique : features (K, N, F) dont la dernière = disponibilité (0/1)."""
    fc, cash = cfg["features"], cfg["data"]["cash_name"]
    ctx = load_context(cfg, px.index)
    feats, context = compute_daily_features(px, ctx, cfg)
    names = asset_feature_names(cfg)
    all_dates = weekly_decision_dates(px.index, fc["weekly_rule"])
    ctx_ok = context.loc[all_dates].notna().all(axis=1)
    avail_all = pd.concat([feats[n].loc[all_dates].notna() for n in names], axis=0).groupby(level=0).all()
    avail_all = avail_all.reindex(all_dates)
    ok = ((avail_all.sum(axis=1) >= fc["min_available_assets"]) & ctx_ok).to_numpy()
    start = int(np.argmax(ok))
    if not ok.any() or not ok[start:].all():  # le warm-up ne doit concerner que le début de l'historique
        raise ValueError("trou inattendu dans les features")
    dates = all_dates[start:]
    avail = avail_all.loc[dates].to_numpy()
    n, k = px.shape[1], len(dates)
    x = np.stack([feats[nm].loc[dates].to_numpy() for nm in names], axis=-1)            # (K, N, Fa)
    c = np.repeat(context.loc[dates].to_numpy()[:, None, :], n, axis=1)                 # (K, N, Fc)
    raw = np.concatenate([x, c], axis=-1)
    raw[~avail] = np.nan                                                                 # titre indisponible
    splits = {s: split_transitions(dates, *cfg["split"][s]) for s in ("train", "validation", "test")}
    for s, ix in splits.items():
        if len(ix) == 0:
            raise ValueError(f"split {s} vide")
    tr = splits["train"]
    mean = np.nanmean(raw[tr], axis=(0, 1))
    std = np.nanstd(raw[tr], axis=(0, 1))
    std = np.where(std > 1e-12, std, 1.0)
    norm = np.clip((raw - mean) / std, -fc["clip"], fc["clip"])
    norm = np.nan_to_num(norm, nan=0.0)
    features = np.concatenate([norm, avail[..., None].astype(float)], axis=-1).astype(np.float32)

    prices = px.copy()
    prices[cash] = (1.0 + cfg["data"]["cash_daily_return"]) ** np.arange(len(px), dtype=float)
    wr = weekly_asset_returns(prices, dates).to_numpy()
    wr[:, :-1] = np.where(avail[:-1], np.nan_to_num(wr[:, :-1], nan=0.0), 0.0)  # titre indisponible : 0 (poids 0)
    if np.isnan(wr).any():
        raise ValueError("rendement hebdomadaire manquant pour un titre disponible")
    bw = weekly_asset_returns(bench, dates).to_numpy()
    f_all = mean.shape[0]
    normalizer = Normalizer(mean=np.broadcast_to(mean, (n, f_all)).copy(), std=np.broadcast_to(std, (n, f_all)).copy(),
                            clip=fc["clip"])
    for s, ix in splits.items():
        logger.info("split %-10s : %4d périodes, %s -> %s, titres disponibles %d -> %d", s, len(ix),
                    dates[ix[0]].date(), dates[ix[-1]].date(), avail[ix[0]].sum(), avail[ix[-1]].sum())
    return MarketData(
        dates=dates, assets=px.columns.tolist(), cash_name=cash, features=features, raw_features=raw,
        weekly_returns=wr, splits=splits, daily_returns=prices.pct_change(fill_method=None).iloc[1:],
        benchmark_weekly=bw, normalizer=normalizer, benchmark_names=[str(b) for b in bench.columns], available=avail,
    )
