"""Retargeter implementations.

Each retargeter wraps an algorithm and implements the BaseRetargeter interface.
Add new algorithms here by:
  1. Creating config_types/retargeters/<name>.py
  2. Creating retargeters/<name>.py implementing BaseRetargeter
  3. Registering in retargeters/registry.py
"""

from holosoma_retargeting.retargeters.base import BaseRetargeter
from holosoma_retargeting.retargeters.registry import RETARGETER_REGISTRY, build_retargeter

__all__ = ["BaseRetargeter", "RETARGETER_REGISTRY", "build_retargeter"]
