"""Port for persisting trained artifacts so a session can skip re-training.

`ArtifactStore` is an abstract port. It lives in the application layer because it speaks in
terms of `TrainedArtifacts` (an application value), keeping the dependency rule intact: the
concrete adapter in `fifa26.persistence` (outer) depends on this interface (inner), never
the reverse. `main.py` wires a concrete store and hands it to the interactive app, which
loads a saved model by default and only re-trains when the user asks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fifa26.application.training import TrainedArtifacts


class ArtifactStore(ABC):
    @abstractmethod
    def exists(self) -> bool:
        """True if a previously saved set of artifacts is available to load."""
        raise NotImplementedError

    @abstractmethod
    def save(self, artifacts: TrainedArtifacts) -> Path:
        """Persist `artifacts` and return the path they were written to."""
        raise NotImplementedError

    @abstractmethod
    def load(self) -> TrainedArtifacts | None:
        """Return the saved artifacts, or `None` if missing/unreadable (so callers re-train)."""
        raise NotImplementedError
