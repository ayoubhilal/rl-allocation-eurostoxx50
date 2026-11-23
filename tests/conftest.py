"""Configuration pytest : racine du projet dans sys.path + fixtures partagées."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()
