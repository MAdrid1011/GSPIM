"""Runnable functional reference for the GSPIM paper mechanisms."""

from .config import ArchitectureConfig
from .pipeline import GSPIMPipeline, SequenceResult, WindowResult
from .runtime import PIMTaskDescriptor, PIMTaskKind, Runtime

__all__ = [
    "ArchitectureConfig",
    "GSPIMPipeline",
    "PIMTaskDescriptor",
    "PIMTaskKind",
    "Runtime",
    "SequenceResult",
    "WindowResult",
]
