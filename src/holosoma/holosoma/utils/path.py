"""Path resolution utilities for package data files."""

from __future__ import annotations

from pathlib import Path


def resolve_data_file_path(file_path: str) -> str:
    """
    Resolve a data file path.

    Handles multiple path formats:
    1. S3 paths: "s3://bucket/path/to/file.npz" -> return as-is
    2. holosoma_data paths: "holosoma_data/..." -> resolved via holosoma_data package
    3. Absolute paths: "/path/to/file.npz" -> returned as-is
    4. Relative paths: "./data/file.npz" or "../data/file.npz" -> resolved relative to CWD

    Args:
        file_path: The path to resolve

    Returns:
        The resolved absolute path as a string

    Examples:
        >>> # holosoma_data package (canonical)
        >>> path = resolve_data_file_path("holosoma_data/pipeline/retargeted/g1_29dof/omniretarget/sub3_largebox_003/retargeted.npz")
        >>> print(path)
        /path/to/holosoma_data/holosoma_data/pipeline/retargeted/...

        >>> # holosoma_data pipeline converted (canonical)
        >>> path = resolve_data_file_path("holosoma_data/pipeline/converted/g1_29dof/whole_body_tracking/motion_crawl_slope.npz")
        >>> print(path)
        /path/to/holosoma_data/holosoma_data/pipeline/converted/g1_29dof/whole_body_tracking/motion_crawl_slope.npz

        >>> # User's custom file (absolute)
        >>> path = resolve_data_file_path("/home/user/my_motions/custom.npz")
        >>> print(path)
        /home/user/my_motions/custom.npz
    """
    # 1. If it's an S3 path, return as-is
    if file_path.startswith("s3://"):
        return file_path

    # 2. holosoma_data package paths
    if file_path.startswith("holosoma_data/") or file_path == "holosoma_data":
        try:
            from holosoma_data import HOLOSOMA_DATA_ROOT
            suffix = file_path[len("holosoma_data"):].lstrip("/")
            return str(HOLOSOMA_DATA_ROOT / suffix) if suffix else str(HOLOSOMA_DATA_ROOT)
        except ImportError:
            pass  # fall through to absolute/relative resolution

    # 3. If it's an absolute path, return as-is
    path_obj = Path(file_path)
    if path_obj.is_absolute():
        return file_path

    # 4. Otherwise, resolve relative path to absolute (relative to CWD)
    resolved = path_obj.resolve()
    return str(resolved)
