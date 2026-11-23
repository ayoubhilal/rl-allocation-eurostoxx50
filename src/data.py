"""Phase 1 : constituants Euro Stoxx 50, téléchargement Yahoo Finance, nettoyage, univers."""

from __future__ import annotations

import io
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import project_path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- constituants
def normalize_name(name: str) -> str:
    """Nom de société normalisé (ascii, minuscules, alphanumérique) pour les rapprochements."""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_name.lower())


def parse_listing(listing: str) -> tuple[str, str] | None:
    """Découpe 'Euronext Paris: AI' en ('Euronext Paris', 'AI')."""
    if not isinstance(listing, str) or ":" not in listing:
        return None
    exchange, symbol = listing.split(":", 1)
    clean = lambda s: re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()  # noqa: E731
    exchange, symbol = clean(exchange), clean(symbol)
    if not exchange or not symbol:
        return None
    return exchange, symbol


def to_yahoo_ticker(exchange: str, symbol: str, suffix_map: dict[str, str]) -> str | None:
    """Convertit (place, symbole local) en ticker Yahoo, ex. ('Nasdaq Helsinki', 'NDA FI') -> 'NDA-FI.HE'."""
    suffix = suffix_map.get(exchange)
    if suffix is None:
        return None
    return symbol.replace(" ", "-").upper() + suffix


def fetch_wikipedia_constituents(cfg: dict[str, Any]) -> pd.DataFrame:
    """Scrape la table de composition Wikipédia ; renvoie name / listing / wiki_ticker."""
    import requests

    dcfg = cfg["data"]
    resp = requests.get(
        dcfg["wikipedia_url"], headers={"User-Agent": dcfg["user_agent"]}, timeout=dcfg["http_timeout"]
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    for table in tables:
        cols = {str(c).strip().lower(): c for c in table.columns}
        if "name" in cols and "main listing" in cols:
            out = pd.DataFrame(
                {
                    "name": table[cols["name"]].astype(str).str.strip(),
                    "listing": table[cols["main listing"]].astype(str),
                    "wiki_ticker": table[cols["ticker"]].astype(str).str.strip() if "ticker" in cols else "",
                }
            )
            logger.info("Wikipédia : table de composition trouvée (%d lignes)", len(out))
            return out
    raise ValueError("Table de composition introuvable sur la page Wikipédia")


def build_candidates(cfg: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Liste de sociétés avec tickers Yahoo candidats (ordre de priorité) ; source = wikipedia|fallback."""
    fallback = pd.read_csv(project_path(cfg["paths"]["fallback_tickers"]))
    fb_by_name = {normalize_name(n): t for n, t in zip(fallback["name"], fallback["yahoo_ticker"])}
    try:
        wiki = fetch_wikipedia_constituents(cfg)
    except Exception as exc:  # réseau, HTML modifié...
        logger.warning("Scraping Wikipédia impossible (%s) -> liste de secours %s", exc, cfg["paths"]["fallback_tickers"])
        cands = pd.DataFrame(
            {"name": fallback["name"], "candidates": [[t] for t in fallback["yahoo_ticker"]]}
        )
        return cands, "fallback"

    rows = []
    for _, r in wiki.iterrows():
        cand: list[str] = []
        parsed = parse_listing(r["listing"])
        if parsed is not None:
            mapped = to_yahoo_ticker(*parsed, cfg["data"]["exchange_suffix"])
            if mapped is not None:
                cand.append(mapped)
            else:
                logger.warning("Place de cotation non mappée '%s' pour %s", parsed[0], r["name"])
        else:
            logger.warning("Cotation illisible pour %s : %r", r["name"], r["listing"])
        if "." in r["wiki_ticker"] and r["wiki_ticker"] not in cand:
            cand.append(r["wiki_ticker"])
        key = normalize_name(r["name"])
        fb = fb_by_name.get(key) or next((t for n, t in fb_by_name.items() if n in key or key in n), None)
        if fb is not None and fb not in cand:
            cand.append(fb)
        if not cand:
            logger.warning("Aucun ticker candidat pour %s : société ignorée", r["name"])
            continue
        rows.append({"name": r["name"], "candidates": cand})
    return pd.DataFrame(rows), "wikipedia"


# --------------------------------------------------------------------------- téléchargement
def download_ohlcv(
    tickers: list[str], start: str | None, end: str | None, cfg: dict[str, Any], period: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Télécharge Close (ajusté, auto_adjust=True) et Volume ; retente les tickers vides."""
    import yfinance as yf

    retries = cfg["data"]["download_retries"]
    close, volume = pd.DataFrame(), pd.DataFrame()
    todo = list(dict.fromkeys(tickers))
    for attempt in range(1, retries + 1):
        if not todo:
            break
        kwargs = dict(auto_adjust=True, progress=False, threads=True, group_by="column")
        if period is not None:
            raw = yf.download(todo, period=period, **kwargs)
        else:
            raw = yf.download(todo, start=start, end=end, **kwargs)
        if raw is None or raw.empty:
            logger.warning("Téléchargement vide (tentative %d/%d)", attempt, retries)
            time.sleep(2 * attempt)
            continue
        c = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].set_axis(todo, axis=1)
        v = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]].set_axis(todo, axis=1)
        ok = [t for t in todo if t in c.columns and c[t].notna().any()]
        close = pd.concat([close, c[ok]], axis=1)
        volume = pd.concat([volume, v[ok]], axis=1)
        todo = [t for t in todo if t not in ok]
        if todo:
            logger.warning("Tentative %d/%d : pas de données pour %s", attempt, retries, todo)
            time.sleep(2 * attempt)
    close.index = pd.to_datetime(close.index).tz_localize(None) if len(close) else close.index
    volume.index = close.index
    return close.sort_index(), volume.sort_index()


def verify_and_resolve(cands: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Vérifie chaque ticker candidat sur un historique court ; retient le premier valide par société."""
    all_tickers = sorted({t for c in cands["candidates"] for t in c})
    close, _ = download_ohlcv(all_tickers, None, None, cfg, period=cfg["data"]["verify_period"])
    valid = set(close.columns)
    rows = []
    for _, r in cands.iterrows():
        tried = r["candidates"]
        chosen = next((t for t in tried if t in valid), None)
        failed = [t for t in tried if t not in valid]
        if failed:
            logger.warning("%s : tickers sans données Yahoo %s", r["name"], failed)
        if chosen is None:
            logger.error("%s : aucun ticker valide, société exclue", r["name"])
        elif chosen != tried[0]:
            logger.info("%s : ticker de repli %s utilisé (au lieu de %s)", r["name"], chosen, tried[0])
        rows.append({"name": r["name"], "candidates": "|".join(tried), "yahoo_ticker": chosen,
                     "failed": "|".join(failed)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- nettoyage
@dataclass
class CleaningResult:
    close: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.DataFrame
    removed: dict[str, str] = field(default_factory=dict)
    dropped_days: int = 0


def clean_prices(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    start_date: str,
    coverage_tolerance_days: int,
    max_missing_frac: float,
    max_ffill_days: int,
) -> CleaningResult:
    """Calendrier commun, filtre de couverture, forward-fill borné, suppression des jours incomplets."""
    benchmark = benchmark.to_frame() if isinstance(benchmark, pd.Series) else benchmark
    calendar = close.index.union(benchmark.dropna(how="all").index)
    close = close.reindex(calendar)
    volume = volume.reindex(calendar)
    benchmark = benchmark.reindex(calendar)
    calendar_end = calendar.max()
    deadline_start = pd.Timestamp(start_date) + pd.Timedelta(days=coverage_tolerance_days)
    deadline_end = calendar_end - pd.Timedelta(days=coverage_tolerance_days)
    removed: dict[str, str] = {}
    for t in close.columns:
        s = close[t]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            removed[t] = "aucune donnée"
        elif first > deadline_start:
            removed[t] = f"historique incomplet : 1re cotation le {first.date()} (> {deadline_start.date()})"
        elif last < deadline_end:
            removed[t] = f"cotation interrompue : dernière cotation le {last.date()}"
        else:
            miss = s.loc[first:].isna().mean()
            if miss > max_missing_frac:
                removed[t] = f"trop de valeurs manquantes ({miss:.1%} > {max_missing_frac:.0%})"
    keep = [t for t in close.columns if t not in removed]
    close, volume = close[keep], volume[keep]
    close = close.ffill(limit=max_ffill_days)
    benchmark = benchmark.ffill(limit=max_ffill_days)
    mask = close.notna().all(axis=1) & benchmark.notna().all(axis=1)
    dropped = int((~mask).sum())
    close, benchmark = close[mask], benchmark[mask]
    volume = volume.loc[close.index].fillna(0.0)
    return CleaningResult(close=close, volume=volume, benchmark=benchmark, removed=removed, dropped_days=dropped)


def select_universe(
    close: pd.DataFrame, volume: pd.DataFrame, n_assets: int, train_start: str, train_end: str
) -> pd.Series:
    """Top-N par valeur moyenne échangée (prix x volume) calculée sur la période d'entraînement seulement."""
    window = slice(pd.Timestamp(train_start), pd.Timestamp(train_end))
    traded = (close.loc[window] * volume.loc[window]).mean(axis=0)
    traded = traded.sort_values(ascending=False, kind="mergesort")
    if len(traded) < n_assets:
        logger.warning("Seulement %d titres disponibles (< N=%d)", len(traded), n_assets)
    return traded.iloc[:n_assets]


def add_cash(prices: pd.DataFrame, daily_return: float, cash_name: str) -> pd.DataFrame:
    """Ajoute un actif sans risque au rendement quotidien constant (prix initial 1)."""
    out = prices.copy()
    out[cash_name] = (1.0 + daily_return) ** np.arange(len(out), dtype=float)
    return out


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements simples journaliers (la 1re ligne, indéfinie, est retirée)."""
    return prices.pct_change(fill_method=None).iloc[1:]


# --------------------------------------------------------------------------- persistance
def processed_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    d = project_path(cfg["paths"]["processed_dir"])
    return {
        "prices": d / "prices.csv",
        "returns": d / "returns.csv",
        "benchmark": d / "benchmark.csv",
        "universe": d / "universe.csv",
        "report": d / "report.md",
    }


def load_processed(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge prix (N actifs + CASH), rendements journaliers et benchmarks (une colonne par benchmark)."""
    p = processed_paths(cfg)
    prices = pd.read_csv(p["prices"], index_col=0, parse_dates=True)
    returns = pd.read_csv(p["returns"], index_col=0, parse_dates=True)
    bench = pd.read_csv(p["benchmark"], index_col=0, parse_dates=True)
    return prices, returns, bench


def write_report(
    path: Path,
    cfg: dict[str, Any],
    source: str,
    resolved: pd.DataFrame,
    cleaning: CleaningResult,
    universe: pd.Series,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    end_date: str,
) -> None:
    """Rapport Markdown de la construction du jeu de données."""
    names = dict(zip(resolved["yahoo_ticker"], resolved["name"]))
    lines = [
        "# Rapport de données",
        "",
        f"- Généré le : {date.today().isoformat()}",
        f"- Source des constituants : **{source}** ({cfg['data']['wikipedia_url']})",
        f"- Téléchargement Yahoo Finance (auto_adjust=True) : {cfg['data']['start_date']} -> {end_date}",
        f"- Benchmarks : {cfg['data']['benchmark']} (indice prix, dividendes non réinvestis) et "
        f"{cfg['data']['benchmark_total_return']} (ETF Euro Stoxx 50, prix ajustés = dividendes inclus, frais ~0,1 %/an)",
        f"- Période couverte après nettoyage : {prices.index.min().date()} -> {prices.index.max().date()}",
        f"- Lignes : prix = {len(prices)}, rendements = {len(returns)} ; colonnes = {prices.shape[1]} "
        f"({len(universe)} titres + {cfg['data']['cash_name']})",
        f"- Jours supprimés (valeurs manquantes après ffill <= {cfg['data']['max_ffill_days']} j) : {cleaning.dropped_days}",
        f"- Actif {cfg['data']['cash_name']} : rendement quotidien constant = {cfg['data']['cash_daily_return']}",
        "",
        "## Résolution des tickers",
        "",
        "| Société | Ticker Yahoo | Candidats | Échecs |",
        "|---|---|---|---|",
    ]
    for _, r in resolved.iterrows():
        lines.append(f"| {r['name']} | {r['yahoo_ticker'] or '**aucun**'} | {r['candidates']} | {r['failed'] or ''} |")
    lines += [
        "",
        f"## Univers retenu (top {cfg['data']['n_assets']} par valeur échangée moyenne, "
        f"train {cfg['split']['train'][0]} -> {cfg['split']['train'][1]})",
        "",
        "| Rang | Ticker | Société | Valeur échangée moyenne (M EUR/jour) |",
        "|---|---|---|---|",
    ]
    for i, (t, v) in enumerate(universe.items(), 1):
        lines.append(f"| {i} | {t} | {names.get(t, '')} | {v / 1e6:,.1f} |")
    lines += ["", "## Titres retirés au nettoyage", ""]
    if cleaning.removed:
        lines += ["| Ticker | Société | Raison |", "|---|---|---|"]
        lines += [f"| {t} | {names.get(t, '')} | {why} |" for t, why in cleaning.removed.items()]
    else:
        lines.append("Aucun.")
    not_selected = [t for t in cleaning.close.columns if t not in universe.index]
    lines += ["", "## Titres valides mais hors univers (liquidité inférieure)", "", ", ".join(not_selected) or "Aucun."]
    unresolved = resolved[resolved["yahoo_ticker"].isna()]
    lines += ["", "## Sociétés sans ticker Yahoo valide", ""]
    lines.append(", ".join(unresolved["name"]) if len(unresolved) else "Aucune.")
    lines += [
        "",
        "## Avertissement : biais de survie",
        "",
        "L'univers part de la composition **actuelle** de l'indice : les sociétés sorties de l'indice "
        "depuis 2009 sont absentes, ce qui biaise les performances historiques à la hausse.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
