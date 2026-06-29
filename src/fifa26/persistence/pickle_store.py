"""Pickle-backed `ArtifactStore`.

`TrainedArtifacts` is plain-pickle-safe: the Bayesian model keeps only floats/dicts (its
PyMC trace is discarded after fitting), the XGBoost model keeps a serialisable booster, and
the rest are DataFrames, dicts and value objects. So a single `pickle.dump` of the whole
artifacts object is enough to reload a fully trained predictor without re-running NUTS.

Reads are best-effort: a missing, corrupt or incompatible file makes `load()` return
`None`, which tells the caller to re-train instead of crashing the session.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from fifa26.application.artifact_store import ArtifactStore
from fifa26.application.training import TrainedArtifacts


class PickleArtifactStore(ArtifactStore):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def exists(self) -> bool:
        return self._path.is_file()

    def save(self, artifacts: TrainedArtifacts) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as fh:
            pickle.dump(artifacts, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return self._path

    def load(self) -> TrainedArtifacts | None:
        if not self.exists():
            return None
        try:
            with self._path.open("rb") as fh:
                artifacts = pickle.load(fh)
        except Exception:  # corrupt / stale / version mismatch -> re-train
            return None
        return artifacts if isinstance(artifacts, TrainedArtifacts) else None
