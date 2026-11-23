"""Entraînement PPO (stable-baselines3) : fabriques d'agents, callback de diagnostic, boucle d'entraînement."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env import PortfolioEnv, rollout
from src.features import MarketData
from src.metrics import metrics_from_frame
from src.policies import SharedAssetPolicy
from src.utils import project_path, set_global_seed

logger = logging.getLogger(__name__)

ACTIVATIONS = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}


# --------------------------------------------------------------------------- agents
class AgentFactory(ABC):
    """Interface commune : construire / charger un agent SB3. Ajouter un algo = ajouter une sous-classe."""

    name: str = ""
    algo_class: type[BaseAlgorithm]

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    @abstractmethod
    def build(self, env: Any, seed: int, tensorboard_log: str | None) -> BaseAlgorithm:
        """Crée un modèle non entraîné."""

    def load(self, path: str | Path) -> BaseAlgorithm:
        return self.algo_class.load(str(path), device="cpu")


class PPOFactory(AgentFactory):
    name = "PPO"
    algo_class = PPO

    def build(self, env: Any, seed: int, tensorboard_log: str | None) -> BaseAlgorithm:
        p = self.cfg["rl"]["ppo"]
        policy: Any
        if p["policy"] == "mlp":
            policy = "MlpPolicy"
            policy_kwargs: dict[str, Any] = dict(
                net_arch=dict(pi=list(p["net_arch"]), vf=list(p["net_arch"])),
                activation_fn=ACTIVATIONS[p["activation"]],
                log_std_init=p["log_std_init"],
            )
        elif p["policy"] == "shared_asset":
            policy = SharedAssetPolicy
            names = all_feature_names(self.cfg)
            policy_kwargs = dict(
                n_assets=int(self.cfg["data"]["n_assets"]), n_features=len(names), masked=names[-1] == "available",
                hidden=int(p["net_arch"][0]), n_layers=len(p["net_arch"]), activation=p["activation"],
                log_std_init=p["log_std_init"],
            )
        else:
            raise ValueError(f"rl.ppo.policy inconnu : {p['policy']}")
        return PPO(
            policy, env, learning_rate=learning_rate_schedule(p), n_steps=p["n_steps"], batch_size=p["batch_size"],
            n_epochs=p["n_epochs"], gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
            ent_coef=p["ent_coef"], vf_coef=p["vf_coef"], max_grad_norm=p["max_grad_norm"],
            policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, seed=seed, device="cpu", verbose=0,
        )


def all_feature_names(cfg: dict[str, Any]) -> list[str]:
    """Noms des features par actif (univers dynamique, dernière feature = disponibilité)."""
    from src.universe import feature_names

    return feature_names(cfg)


def learning_rate_schedule(p: dict[str, Any]) -> float | Callable[[float], float]:
    """Taux constant, ou décroissance linéaire (progress_remaining va de 1 à 0)."""
    lr = float(p["learning_rate"])
    if p["lr_schedule"] == "constant":
        return lr
    if p["lr_schedule"] == "linear":
        return lambda progress_remaining: lr * progress_remaining
    raise ValueError(f"lr_schedule inconnu : {p['lr_schedule']}")


class _NotImplementedFactory(AgentFactory):
    def build(self, env: Any, seed: int, tensorboard_log: str | None) -> BaseAlgorithm:
        raise NotImplementedError(
            f"{self.name} est prévu mais pas encore implémenté : ajouter une section rl.{self.name.lower()} "
            f"dans config.yaml et construire {self.name}(...) ici sur le modèle de PPOFactory."
        )


class A2CFactory(_NotImplementedFactory):
    name = "A2C"


class SACFactory(_NotImplementedFactory):
    name = "SAC"


class TD3Factory(_NotImplementedFactory):
    name = "TD3"


ALGORITHMS: dict[str, type[AgentFactory]] = {f.name: f for f in (PPOFactory, A2CFactory, SACFactory, TD3Factory)}


def get_factory(cfg: dict[str, Any]) -> AgentFactory:
    return ALGORITHMS[cfg["rl"]["algorithm"]](cfg)


# --------------------------------------------------------------------------- évaluation / callback
def evaluate_policy_on_split(model: BaseAlgorithm, md: MarketData, split: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Parcours déterministe d'un split avec la politique (action moyenne)."""
    env = PortfolioEnv(md, split, cfg, mode="eval")
    return rollout(env, lambda obs: model.predict(obs, deterministic=True)[0])


class ValidationSharpeCallback(BaseCallback):
    """Évalue le Sharpe de validation tous les `eval_freq` pas.

    checkpoint_selection = "best_validation" : sauve le meilleur et arrête après `patience` évaluations sans progrès.
    checkpoint_selection = "final" : historique de diagnostic uniquement (aucune sélection, aucun arrêt).
    """

    def __init__(self, md: MarketData, cfg: dict[str, Any], save_path: Path) -> None:
        super().__init__()
        r = cfg["rl"]
        self.md, self.cfg, self.save_path = md, cfg, save_path
        self.select = r["checkpoint_selection"] == "best_validation"
        self.eval_freq, self.patience, self.min_evals = r["eval_freq"], r["early_stopping_patience"], r["min_evals_before_stop"]
        self.best_sharpe, self.best_step, self.no_improve = -np.inf, 0, 0
        self.history: list[dict[str, Any]] = []
        self.stopped_early = False

    def _evaluate(self, eligible: bool = True) -> bool:
        df = evaluate_policy_on_split(self.model, self.md, "validation", self.cfg)
        m = metrics_from_frame(df, self.cfg)
        sharpe = m["sharpe"] if np.isfinite(m["sharpe"]) else -np.inf
        ep = [e["r"] for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
        row = {"timesteps": self.num_timesteps, "eligible": eligible, "val_sharpe": m["sharpe"], "val_ann_return": m["ann_return"],
               "val_ann_vol": m["ann_vol"], "val_max_drawdown": m["max_drawdown"], "val_avg_turnover": m["avg_turnover"],
               "train_ep_reward_mean": float(np.mean(ep)) if ep else np.nan}
        self.history.append(row)
        for key in ("val_sharpe", "val_ann_return", "val_avg_turnover"):
            self.logger.record(f"validation/{key[4:]}", row[key])
        if not self.select:
            logger.info("[%s] pas %d : Sharpe val %.3f (diagnostic, pas de sélection), récompense moy. train %.2f",
                        self.save_path.name, self.num_timesteps, m["sharpe"], row["train_ep_reward_mean"])
            return True
        if not eligible:  # politique initiale : référence seulement, jamais sélectionnée
            logger.info("[%s] politique initiale (non entraînée) : Sharpe val %.3f", self.save_path.name, m["sharpe"])
            return True
        if sharpe > self.best_sharpe:
            self.best_sharpe, self.best_step, self.no_improve = sharpe, self.num_timesteps, 0
            self.model.save(self.save_path / "best_model")
        else:
            self.no_improve += 1
        logger.info("[%s] pas %d : Sharpe val %.3f (meilleur %.3f @ %d), récompense moy. train %.2f",
                    self.save_path.name, self.num_timesteps, m["sharpe"], self.best_sharpe, self.best_step,
                    row["train_ep_reward_mean"])
        if len(self.history) >= self.min_evals and self.no_improve >= self.patience:
            logger.info("[%s] early stopping au pas %d", self.save_path.name, self.num_timesteps)
            self.stopped_early = True
            return False
        return True

    def _on_training_start(self) -> None:
        self._evaluate(eligible=False)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            return self._evaluate()
        return True


# --------------------------------------------------------------------------- entraînement
def final_checkpoint_name(cfg: dict[str, Any]) -> str:
    return "best_model.zip" if cfg["rl"]["checkpoint_selection"] == "best_validation" else "final_model.zip"


def train_agent(cfg: dict[str, Any], md: MarketData, seed: int, out: Path, tb_name: str) -> dict[str, Any]:
    """Entraîne un agent sur md.splits['train'] (callback sur md.splits['validation'])."""
    torch.set_num_threads(cfg["rl"]["torch_threads"])
    set_global_seed(seed)
    set_random_seed(seed)
    algo = cfg["rl"]["algorithm"]
    out.mkdir(parents=True, exist_ok=True)

    n_envs = int(cfg["rl"]["n_envs"])
    train_env = DummyVecEnv([
        (lambda i=i: Monitor(PortfolioEnv(md, "train", cfg, "train"), str(out / ("monitor" if i == 0 else f"monitor_{i}"))))
        for i in range(n_envs)
    ])
    train_env.seed(seed)
    tb_dir = project_path(cfg["paths"]["tensorboard_dir"])
    model = get_factory(cfg).build(train_env, seed, str(tb_dir))
    cb = ValidationSharpeCallback(md, cfg, out)
    t0 = time.time()
    model.learn(total_timesteps=cfg["rl"]["total_timesteps"], callback=cb, tb_log_name=tb_name,
                reset_num_timesteps=True)
    duration = time.time() - t0
    model.save(out / "final_model")

    hist = pd.DataFrame(cb.history)
    hist.to_csv(out / "eval_history.csv", index=False)
    monitor = pd.read_csv(out / "monitor.monitor.csv", skiprows=1)
    summary = {
        "algorithm": algo,
        "seed": seed,
        "timesteps_done": int(model.num_timesteps),
        "checkpoint_selection": cfg["rl"]["checkpoint_selection"],
        "best_val_sharpe": float(cb.best_sharpe) if cb.select else float("nan"),
        "final_val_sharpe": float(cb.history[-1]["val_sharpe"]) if cb.history else float("nan"),
        "best_step": int(cb.best_step),
        "stopped_early": cb.stopped_early,
        "n_episodes": int(len(monitor)),
        "train_reward_mean_all": float(monitor["r"].mean()),
        "train_reward_mean_last100": float(monitor["r"].tail(100).mean()),
        "duration_sec": round(duration, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("[%s] terminé en %.0fs (%d pas) : Sharpe val final %.3f", out.name, duration, model.num_timesteps,
                summary["final_val_sharpe"])
    return summary

