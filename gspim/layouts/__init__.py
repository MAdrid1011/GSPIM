"""Paper-scoped representation adapters."""

from .anchored import Anchored4DGSLayout
from .ex4dgs import Ex4DGSLayout
from .explicit4d import Explicit4DLayout

__all__ = ["Anchored4DGSLayout", "Ex4DGSLayout", "Explicit4DLayout"]
