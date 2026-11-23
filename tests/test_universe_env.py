"""Tests du modèle : univers dynamique (entrées progressives), signaux sans look-ahead, masque de disponibilité
dans l'environnement, les baselines et le réseau partagé, portefeuille de référence des inclinaisons."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src.baselines import base_weight_path, run_equal_weight, run_max_sharpe
from src.env import PortfolioEnv, rollout
from src.train import get_factory
from src.universe import compute_daily_features, masked_rank, feature_names
from src.utils import load_config


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """Univers synthétique de 8 titres dont 2 entrent tardivement, contexte de marché synthétique."""
    c = load_config()
    c["data"]["n_assets"] = 8
    c["features"]["min_available_assets"] = 4
    c["split"] = {"train": ["2010-01-01", "2011-12-31"], "validation": ["2012-01-01", "2012-12-31"],
                  "test": ["2013-01-01", None]}
    idx = pd.bdate_range("2009-01-01", "2013-12-31")
    rng = np.random.default_rng(0)
    px = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (len(idx), 8)), axis=0)), idx,
                      [f"S{i}" for i in range(8)])
    px.loc[:"2010-06-30", "S6"] = np.nan  # entrée mi-2010
    px.loc[:"2011-09-30", "S7"] = np.nan  # entrée fin 2011
    ctx = pd.DataFrame({"^STOXX50E": px.iloc[:, :6].mean(axis=1) * 60, "^VIX": 20 + np.sin(np.arange(len(idx)) / 30)}, idx)
    return c, px, ctx


@pytest.fixture(scope="module")
def md(setup, monkeypatch_module):
    c, px, ctx = setup
    import src.universe as u

    monkeypatch_module.setattr(u, "load_context", lambda cfg, cal: ctx.reindex(cal))
    bench = pd.DataFrame({"B": px.iloc[:, :6].mean(axis=1)})
    return u.build_market_data(px, bench, c)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_masked_rank() -> None:
    df = pd.DataFrame([[1.0, 3.0, np.nan, 2.0]])
    np.testing.assert_allclose(masked_rank(df).iloc[0].to_numpy(), [-1.0, 1.0, np.nan, 0.0])


def test_signals_no_lookahead(setup) -> None:
    c, px, ctx = setup
    feats, context = compute_daily_features(px, ctx, c)
    t = pd.Timestamp("2012-03-15")
    shocked = px.copy()
    shocked.loc[shocked.index > t] *= np.random.default_rng(1).uniform(0.5, 2, shocked.loc[shocked.index > t].shape)
    ctx2 = ctx.copy()
    ctx2.loc[ctx2.index > t] *= 1.7
    f2, c2 = compute_daily_features(shocked, ctx2, c)
    for n in feats:
        pd.testing.assert_frame_equal(feats[n].loc[:t], f2[n].loc[:t])
    pd.testing.assert_frame_equal(context.loc[:t], c2.loc[:t])
    # momentum 12-1 = prix il y a 1 mois / prix il y a 12 mois - 1
    k = 400
    s, lb = c["features"]["signal_skip_days"], c["features"]["signal_windows_days"]["mom_12"]
    raw = px["S0"].iloc[k - s] / px["S0"].iloc[k - lb] - 1
    ranks = (px.shift(s) / px.shift(lb) - 1).iloc[k]
    assert np.isclose(masked_rank(ranks.to_frame().T).iloc[0]["S0"], feats["rank_mom_12_1"].iloc[k]["S0"])
    assert np.isfinite(raw)


def test_dynamic_availability(md, setup) -> None:
    c, px, _ = setup
    j7 = md.assets.index("S7")
    first = md.dates[np.argmax(md.available[:, j7])]
    # investissable seulement après 1 an de cotation (252 jours de bourse : fenêtre du momentum 12 mois)
    first_price = px["S7"].first_valid_index()
    lb = c["features"]["signal_windows_days"]["mom_12"]
    assert px.index.get_loc(first) - px.index.get_loc(first_price) >= lb
    assert (md.features[~md.available][:, -1] == 0).all() and (md.features[md.available][:, -1] == 1).all()
    assert (md.features[~md.available][:, :-1] == 0).all()  # features neutres pour un titre indisponible
    assert md.features.shape[2] == len(feature_names(c))
    assert not np.isnan(md.weekly_returns).any()


def test_env_respects_mask(md, setup) -> None:
    c, _, _ = setup
    env = PortfolioEnv(md, "train", c, "train")
    rng = np.random.default_rng(0)
    for s in range(5):
        env.reset(seed=s)
        for _ in range(20):
            k = env.current_k
            _, _, _, _, info = env.step(rng.uniform(-1, 1, md.n_assets + 1))
            w = info["weights"][:-1]
            assert (w[~md.available[k]] < 1e-12).all() and np.isclose(info["weights"].sum(), 1)
    ev = PortfolioEnv(md, "train", c, "eval")
    obs, _ = ev.reset()
    k0 = ev.current_k
    w0 = obs[-(md.n_assets + 1):]
    np.testing.assert_allclose(w0[:-1], md.available[k0] / md.available[k0].sum(), atol=1e-6)


def test_baselines_respect_mask(md, setup) -> None:
    c, _, _ = setup
    for fn in (run_equal_weight, run_max_sharpe):
        df = fn(md, "train", c)
        w = df[[f"w_{a}" for a in md.assets]].to_numpy()
        avail = md.available[md.splits["train"]]
        assert (w[~avail] < 1e-12).all()
    ew = run_equal_weight(md, "train", c)
    k = md.splits["train"][-1]
    np.testing.assert_allclose(ew[[f"w_{a}" for a in md.assets]].iloc[-1].to_numpy(), md.available[k] / md.available[k].sum())


def test_reference_portfolio(md, setup) -> None:
    c, _, _ = setup
    c2 = copy.deepcopy(c)
    c2["env"]["base_portfolio"] = "maxsharpe"
    c2["env"]["smoothing_alpha"] = 1.0
    env = PortfolioEnv(md, "validation", c2, "eval")
    env.reset()
    k = env.current_k
    base = base_weight_path(md, md.splits["validation"], c2, "maxsharpe")[k]
    w = env.action_to_weights(np.zeros(md.n_assets + 1))  # action nulle = référence (+ plancher)
    m = md.available[k]
    expected = np.zeros(md.n_assets)
    expected[m] = (base[m] + c2["env"]["base_floor"]) / (base[m] + c2["env"]["base_floor"]).sum()
    np.testing.assert_allclose(w[:-1], expected, atol=1e-9)
    assert w[:-1].max() <= c2["baselines"]["max_weight_sharpe"] + 0.05


def test_masked_shared_policy(md, setup) -> None:
    c, _, _ = setup
    env = PortfolioEnv(md, "train", c, "train")
    pol = get_factory(c).build(env, 0, None).policy
    assert pol.mlp_extractor.masked
    obs, _ = env.reset(seed=0)
    n, f = md.n_assets, md.features.shape[2]
    feats = obs[: n * f].reshape(n, f).copy()
    j = int(np.argmin(feats[:, -1])) if (feats[:, -1] == 0).any() else None
    if j is None:
        pytest.skip("tous les titres disponibles à cette date")
    feats2 = feats.copy()
    feats2[j, :-1] = 3.0  # modifier un titre INDISPONIBLE ne doit pas changer le contexte ni les autres scores
    o2 = np.concatenate([feats2.ravel(), obs[n * f:]]).astype(np.float32)
    with torch.no_grad():
        a = pol.get_distribution(torch.as_tensor(obs[None])).distribution.mean[0].numpy()
        b = pol.get_distribution(torch.as_tensor(o2[None])).distribution.mean[0].numpy()
    keep = np.arange(n + 1) != j
    np.testing.assert_allclose(a[keep], b[keep], atol=1e-6)


def test_noise_never_touches_availability_flag(md, setup) -> None:
    c, _, _ = setup
    assert c["env"]["feature_noise"] > 0
    env = PortfolioEnv(md, "train", c, "train")
    n, f = md.n_assets, md.features.shape[2]
    for s in range(3):
        obs, _ = env.reset(seed=s)
        flags = obs[: n * f].reshape(n, f)[:, -1]
        assert set(np.unique(flags)) <= {0.0, 1.0}
        assert np.array_equal(np.sort(flags), np.sort(md.available[env.current_k].astype(np.float32)))
