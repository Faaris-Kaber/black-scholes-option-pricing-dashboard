"""Shared pytest configuration for the Black-Scholes dashboard test suite.

Two jobs, both of which must happen BEFORE any test module is imported:

1. ``sys.path`` bootstrap.  ``src/grid.py`` imports ``from src.black_scholes
   import ...`` (absolute), so the REPO ROOT - the directory that *contains*
   ``src/`` - has to be on ``sys.path``.  Inserting it here makes ``pytest``
   work from any working directory, not just the repo root.

2. Matplotlib backend.  The plotting tests build real figures, so matplotlib
   is forced onto the headless "Agg" backend before ``pyplot`` is imported
   anywhere. Without this the suite
   would depend on a display being present.

Neither step mutates anything under ``src/``; the engine stays pure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# (2) Select a headless backend before any test imports matplotlib.pyplot.
matplotlib.use("Agg", force=True)

# (1) Repo root on sys.path so that "import src.<module>" resolves.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402  (must follow the sys.path bootstrap)
import pytest  # noqa: E402

#: Master seed for every randomized test.  The suite must be deterministic:
#: an unseeded generator turns a real pricing bug into an unreproducible
#: flake, which is worse than no test at all.
MASTER_SEED: int = 20260914

#: The textbook reference case used by spec rows T-2 and T-3.
ATM_CASE: dict[str, float] = {
    "S": 100.0,
    "K": 100.0,
    "T": 1.0,
    "r": 0.05,
    "sigma": 0.20,
}


@pytest.fixture
def rng() -> np.random.Generator:
    """A SEEDED NumPy generator - identical draws on every run and machine."""
    return np.random.default_rng(MASTER_SEED)


@pytest.fixture
def atm() -> dict[str, float]:
    """The spec's reference at-the-money case (S=K=100, T=1, r=5%, sigma=20%)."""
    return dict(ATM_CASE)
