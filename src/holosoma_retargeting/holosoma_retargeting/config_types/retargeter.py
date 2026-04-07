"""Backward-compatibility shim.

RetargeterConfig is now OmniRetargeterConfig.
Import from config_types/retargeters/omniretarget.py for new code.
"""

from holosoma_retargeting.config_types.retargeters.omniretarget import OmniRetargeterConfig as RetargeterConfig

__all__ = ["RetargeterConfig"]
