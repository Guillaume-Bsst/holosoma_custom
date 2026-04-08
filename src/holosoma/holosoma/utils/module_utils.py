import os

import holosoma.utils


def get_holosoma_root() -> str:
    # Use holosoma.utils since holosoma sometimes returns a namespaced module, and its __file__ is None
    return os.path.dirname(os.path.dirname(holosoma.utils.__file__))


def get_holosoma_data_root() -> str:
    """Return the absolute path to the holosoma_data package root."""
    try:
        from holosoma_data import HOLOSOMA_DATA_ROOT
        return str(HOLOSOMA_DATA_ROOT)
    except ImportError:
        raise RuntimeError(
            "holosoma_data is not installed. Run: pip install -e src/holosoma_data"
        )


def resolve_asset_root(asset_root: str) -> str:
    """Resolve an asset_root string that may use @-prefixed aliases.

    Supported aliases:
      @holosoma_data  -> holosoma_data package root  (canonical)
      @holosoma       -> holosoma package root (for non-data assets)

    Args:
        asset_root: raw asset_root string from a config.

    Returns:
        Resolved absolute path string.
    """
    if asset_root.startswith("@holosoma_data/"):
        suffix = asset_root[len("@holosoma_data/"):]
        return os.path.join(get_holosoma_data_root(), suffix)
    if asset_root.startswith("@holosoma_data"):
        return get_holosoma_data_root()
    if asset_root.startswith("@holosoma/"):
        return asset_root.replace("@holosoma", get_holosoma_root())
    return asset_root
