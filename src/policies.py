"""Politique « réseau partagé entre actifs » (type EIIE, Jiang et al. 2017) pour stable-baselines3.

Un même petit MLP note chaque titre à partir de ses propres features et de son poids courant :
l'acteur est équivariant par permutation des actifs (permuter les titres permute les actions),
et le nombre de paramètres ne dépend pas du nombre d'actifs. Le cash et la valeur utilisent
la moyenne des représentations des titres (information de marché).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}


class SharedAssetNet(nn.Module):
    """Extracteur acteur/critique : latent acteur = un score par actif (N+1), latent critique = `hidden`."""

    def __init__(self, n_assets: int, n_features: int, hidden: int, n_layers: int, activation: str,
                 masked: bool = False) -> None:
        super().__init__()
        self.n, self.f = n_assets, n_features
        self.masked = masked  # dernière feature = disponibilité (0/1) : moyenne de contexte sur les titres disponibles
        act = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        d = n_features + 1  # features de l'actif + son poids courant
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), act()]
            d = hidden
        self.embed = nn.Sequential(*layers)
        self.score = nn.Linear(hidden, 1)              # score de chaque titre (partagé)
        self.cash_head = nn.Linear(hidden + 1, 1)      # cash : contexte de marché + poids cash
        self.critic = nn.Sequential(nn.Linear(hidden + 1, hidden), act())
        self.latent_dim_pi = n_assets + 1
        self.latent_dim_vf = hidden

    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, nf = obs.shape[0], self.n * self.f
        feats = obs[:, :nf].reshape(b, self.n, self.f)
        w = obs[:, nf:]
        x = torch.cat([feats, w[:, : self.n].unsqueeze(-1)], dim=-1)
        e = self.embed(x)                              # (B, N, hidden)
        if self.masked:
            m = feats[:, :, -1:]
            pooled = (e * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            pooled = e.mean(dim=1)
        context = torch.cat([pooled, w[:, self.n:]], dim=-1)  # (B, hidden + 1)
        return e, context

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        e, context = self._encode(obs)
        return torch.cat([self.score(e).squeeze(-1), self.cash_head(context)], dim=-1)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        _, context = self._encode(obs)
        return self.critic(context)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(obs), self.forward_critic(obs)


class SharedAssetPolicy(ActorCriticPolicy):
    """ActorCriticPolicy dont l'acteur est SharedAssetNet ; la tête d'action est l'identité."""

    def __init__(self, observation_space: Any, action_space: Any, lr_schedule: Any, n_assets: int, n_features: int,
                 hidden: int = 32, n_layers: int = 2, activation: str = "tanh", masked: bool = False,
                 **kwargs: Any) -> None:
        self.n_assets, self.n_features, self.masked = n_assets, n_features, masked
        self.hidden, self.n_layers, self.activation = hidden, n_layers, activation
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SharedAssetNet(self.n_assets, self.n_features, self.hidden, self.n_layers, self.activation,
                                            self.masked)

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        # les scores par actif SONT les moyennes d'action : pas de couche linéaire non partagée
        self.action_net = nn.Identity()
        for head in (self.mlp_extractor.score, self.mlp_extractor.cash_head):
            nn.init.orthogonal_(head.weight, gain=0.01)  # politique initiale ~ action nulle (= 1/N en mode residual)
            nn.init.zeros_(head.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(n_assets=self.n_assets, n_features=self.n_features, hidden=self.hidden,
                    n_layers=self.n_layers, activation=self.activation, masked=self.masked)
        return data


def count_parameters(module: nn.Module) -> int:
    return int(sum(np.prod(p.shape) for p in module.parameters()))
