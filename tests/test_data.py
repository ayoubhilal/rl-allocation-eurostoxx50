"""Tests hors-ligne de src/data.py (données synthétiques)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import add_cash, clean_prices, compute_returns, parse_listing, select_universe, to_yahoo_ticker

SUFFIX = {"Euronext Paris": ".PA", "FWB": ".DE", "Nasdaq Helsinki": ".HE"}


def test_parse_listing_and_mapping() -> None:
    assert parse_listing("Euronext Paris:\xa0AI") == ("Euronext Paris", "AI")
    assert to_yahoo_ticker(*parse_listing("Nasdaq Helsinki: NDA FI"), SUFFIX) == "NDA-FI.HE"
    assert to_yahoo_ticker("FWB", "muv2", SUFFIX) == "MUV2.DE"
    assert to_yahoo_ticker("Unknown Exchange", "X", SUFFIX) is None
    assert parse_listing("nan") is None


def _synthetic() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    idx = pd.bdate_range("2009-01-02", periods=300)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 4)), axis=0)), idx, list("ABCD"))
    volume = pd.DataFrame(1e6, idx, list("ABCD"))
    bench = pd.Series(1000.0, idx)
    close.loc[idx[:60], "B"] = np.nan          # B commence trop tard
    close.loc[idx[100:102], "A"] = np.nan      # trou de 2 jours -> ffill
    close.loc[idx[200:205], "C"] = np.nan      # trou de 5 jours -> 2 jours supprimés
    return close, volume, bench


def test_clean_prices() -> None:
    close, volume, bench = _synthetic()
    res = clean_prices(close, volume, bench, "2009-01-01", 10, 0.05, 3)
    assert "B" in res.removed and "B" not in res.close.columns
    assert res.close.notna().all().all()
    assert res.dropped_days == 2
    assert res.close.loc[close.index[101], "A"] == close.loc[close.index[99], "A"]
    assert res.close.index.equals(res.volume.index) and res.close.index.equals(res.benchmark.index)


def test_select_universe_uses_train_only() -> None:
    close, volume, _ = _synthetic()
    close = close.ffill().bfill()
    volume.loc[:, "D"] = 5e6
    sel = select_universe(close, volume, 2, "2009-01-01", "2009-06-30")
    vol2 = volume.copy()
    vol2.loc["2009-07-01":, "A"] = 1e12  # volume futur énorme : ne doit rien changer
    assert list(select_universe(close, vol2, 2, "2009-01-01", "2009-06-30").index) == list(sel.index)
    assert sel.index[0] == "D"


def test_cash_and_returns() -> None:
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = add_cash(pd.DataFrame({"A": [1.0, 2, 2, 1, 1]}, idx), 0.001, "CASH")
    rets = compute_returns(prices)
    assert np.allclose(rets["CASH"], 0.001)
    assert np.allclose(rets["A"], [1.0, 0.0, -0.5, 0.0])
