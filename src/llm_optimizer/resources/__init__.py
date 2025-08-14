"""Resource management utilities for LLM optimizer.

This package provides centralized abstractions for GPU resource management
and model memory calculations, eliminating code duplication across the codebase.
"""

from .gpu_manager import GPUResourceManager
from .memory_calculator import ModelMemoryCalculator
from .types import GPUResources, MemoryBreakdown, MemoryLimits

__all__ = [
    "GPUResourceManager",
    "GPUResources",
    "ModelMemoryCalculator",
    "MemoryBreakdown",
    "MemoryLimits"
]
