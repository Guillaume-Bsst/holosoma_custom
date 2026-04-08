"""Single-sequence retargeting entry point.

Usage examples:

    # OmniRetarget (default) — robot_only with SMPLH data
    python examples/robot_retarget.py \
        --task-type robot_only \
        --data_path holosoma_data/datasets/OMOMO \
        --task-name sub3_largebox_003 \
        --data_format smplh \
        --retargeter.debug

    # OmniRetarget — object interaction
    python examples/robot_retarget.py \
        --task-type object_interaction \
        --data_path holosoma_data/datasets/OMOMO \
        --task-name sub3_largebox_003 \
        --data_format smplh \
        --retargeter.visualize

    # GMR — robot_only with SMPLX data
    python examples/robot_retarget.py \
        --retargeter-method gmr \
        --task-type robot_only \
        --data_path holosoma_data/datasets/SFU_demo \
        --task-name my_sequence \
        --data_format smplx \
        --gmr.src_human smplx

    # 27-DOF G1
    python examples/robot_retarget.py \
        --robot-config.robot-dof 27 \
        --task-type robot_only \
        --data_path holosoma_data/datasets/OMOMO \
        --task-name sub3_largebox_003

See holosoma_retargeting/pipeline/run.py for the full pipeline logic.
See holosoma_retargeting/retargeters/registry.py to add new algorithms.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.retargeting import RetargetingConfig  # noqa: E402
from holosoma_retargeting.pipeline.run import run_retargeting  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if __name__ == "__main__":
    cfg = tyro.cli(RetargetingConfig)
    run_retargeting(cfg)
