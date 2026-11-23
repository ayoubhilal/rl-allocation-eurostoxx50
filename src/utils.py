"""Utilitaires transverses : configuration, chemins, logging, seeds."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: Path = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Charge le fichier YAML de configuration."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str | Path) -> Path:
    """Résout un chemin relatif à la racine du projet."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Crée tous les répertoires déclarés dans cfg['paths']."""
    for key, rel in cfg["paths"].items():
        if key.endswith("_dir"):
            project_path(rel).mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: dict[str, Any], name: str, mode: str = "w") -> logging.Logger:
    """Configure un logging console + fichier (results/logs/<name>.log ; mode "a" pour conserver l'historique)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    log_dir = project_path(cfg["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, cfg.get("logging", {}).get("level", "INFO"))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(log_dir / f"{name}.log", mode=mode, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file_handler)
    for noisy in ("yfinance", "urllib3", "peewee", "matplotlib", "curl_cffi"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def set_global_seed(seed: int) -> None:
    """Fixe les seeds de random, numpy et (si disponible) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
