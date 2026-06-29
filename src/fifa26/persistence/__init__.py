"""Persistence adapters: concrete `ArtifactStore` implementations (outer layer)."""
from fifa26.persistence.pickle_store import PickleArtifactStore

__all__ = ["PickleArtifactStore"]
