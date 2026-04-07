"""Retargeter factory registry.

To add a new retargeting algorithm:
  1. Create config_types/retargeters/<name>.py
  2. Create retargeters/<name>.py implementing BaseRetargeter
  3. Add an entry to RETARGETER_REGISTRY below
"""

from __future__ import annotations

from types import SimpleNamespace

from holosoma_retargeting.retargeters.base import BaseRetargeter


# Registry mapping method name → retargeter class.
# Import lazily inside build_retargeter to avoid mandatory heavy dependencies
# (e.g. GMR) when only OmniRetarget is used.
RETARGETER_REGISTRY: dict[str, str] = {
    "omniretarget": "holosoma_retargeting.retargeters.omniretarget.OmniRetargeter",
    "gmr": "holosoma_retargeting.retargeters.gmr.GMRRetargeter",
}


def build_retargeter(method: str, cfg, constants: SimpleNamespace) -> BaseRetargeter:
    """Factory: instantiate the right retargeter from its method name.

    Args:
        method: Retargeter method name, must be a key in RETARGETER_REGISTRY.
        cfg: Algorithm-specific config dataclass (e.g. OmniRetargeterConfig).
        constants: SimpleNamespace with robot/task constants from the pipeline.

    Returns:
        Initialised BaseRetargeter instance.

    Raises:
        ValueError: If method is not registered.
        ImportError: If the retargeter's optional dependencies are missing.
    """
    if method not in RETARGETER_REGISTRY:
        available = list(RETARGETER_REGISTRY.keys())
        raise ValueError(
            f"Unknown retargeter method '{method}'. "
            f"Available: {available}. "
            f"Register new methods in retargeters/registry.py."
        )

    # Lazy import to avoid loading heavy deps when unused
    import importlib

    dotted = RETARGETER_REGISTRY[method]
    module_path, class_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_path)
    retargeter_cls: type[BaseRetargeter] = getattr(module, class_name)

    return retargeter_cls.from_config(cfg, constants)
