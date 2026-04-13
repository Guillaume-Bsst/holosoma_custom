from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import torch
import tyro
from human_body_prior.body_model.body_model import BodyModel  # type: ignore[import-not-found]
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# SMPLH model
# ---------------------------------------------------------------------------

def prep_smplh_model(smplh_root_folder: str) -> dict:
    """Load SMPLH BodyModel instances for male, female and neutral genders."""
    num_betas = 16
    bm_dict = {}
    for gender in ("male", "female", "neutral"):
        bm_fname = os.path.join(smplh_root_folder, gender, "model.npz")
        bm_dict[gender] = BodyModel(bm_fname=bm_fname, num_betas=num_betas)
    return bm_dict


def compute_height(bm_dict: dict, betas: torch.Tensor, gender: str) -> float:
    """Compute subject height from SMPLH T-pose vertices."""
    bm = bm_dict[gender]
    T = 1
    with torch.no_grad():
        out = bm(
            trans=torch.zeros(T, 3),
            root_orient=torch.zeros(T, 3),
            pose_body=torch.zeros(T, 63),
            betas=betas.reshape(1, -1).expand(T, -1),
        )
    verts = out.v.squeeze(0).detach().cpu().numpy()
    return float(verts[:, 1].max() - verts[:, 1].min())


# ---------------------------------------------------------------------------
# Object assets generation
# ---------------------------------------------------------------------------

_URDF_TEMPLATE = dedent("""\
    <?xml version="1.0" ?>
    <robot name="{name}">
      <dynamics damping="0.5" friction="0.9"/>
      <link name="{name}_link">
        <inertial>
          <mass value="0.1"/>
          <origin xyz="0 0 0"/>
          <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
        </inertial>
        <contact>
          <lateral_friction value="0.9"/>
          <rolling_friction value="0.5"/>
          <stiffness value="30000"/>
          <damping value="1000"/>
        </contact>
        <visual>
          <origin rpy="0 0 0" xyz="0 0 0"/>
          <geometry>
            <mesh filename="{name}.obj" scale="1.0 1.0 1.0"/>
          </geometry>
          <material name="mat">
            <color rgba="0.7 0.8 0.9 0.7"/>
          </material>
        </visual>
        <collision name="{name}">
          <origin rpy="0 0 0" xyz="0 0 0"/>
          <geometry>
            <mesh filename="{name}.obj" scale="1.0 1.0 1.0"/>
          </geometry>
        </collision>
      </link>
    </robot>
""")

_XML_TEMPLATE = dedent("""\
    <mujoco model="{name}">
        <compiler meshdir="./"/>

        <default>
            <geom friction="0.9 0.5 0.5" rgba="0.7 0.8 0.9 0.7"/>
            <joint damping="0.5"/>
        </default>

        <asset>
            <mesh name="{name}_mesh" file="{name}.obj" scale="1 1 1"/>
        </asset>

        <worldbody>
            <body name="{name}_link">
                <inertial pos="0 0 0" mass="0.1" diaginertia="0.002 0.002 0.002"/>
                <geom name="{name}" type="mesh" mesh="{name}_mesh"
                      pos="0 0 0" quat="1 0 0 0"
                      rgba="0.7 0.8 0.9 0.7"
                      friction="0.9 0.5 0.5"
                      solref="0.02 1"
                      solimp="0.9 0.95 0.001"/>
            </body>
        </worldbody>
    </mujoco>
""")


def generate_object_assets(captured_objects_dir: str, objects_output_dir: str) -> None:
    """
    For each .obj in captured_objects_dir, create a subfolder in objects_output_dir
    with the .obj + generated .urdf + .xml.

    Skips objects that already exist in objects_output_dir.
    """
    obj_files = sorted(Path(captured_objects_dir).glob("*.obj"))
    if not obj_files:
        print(f"  No .obj files found in {captured_objects_dir}")
        return

    for obj_file in obj_files:
        # Derive clean name: strip suffixes like _cleaned_simplified
        raw_name = obj_file.stem  # e.g. "largebox_cleaned_simplified"
        name = raw_name.replace("_cleaned_simplified", "").replace("_cleaned", "")

        out_dir = Path(objects_output_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy .obj with clean name
        dest_obj = out_dir / f"{name}.obj"
        if not dest_obj.exists():
            shutil.copy(str(obj_file), str(dest_obj))

        # Generate .urdf
        dest_urdf = out_dir / f"{name}.urdf"
        if not dest_urdf.exists():
            dest_urdf.write_text(_URDF_TEMPLATE.format(name=name))

        # Generate .xml
        dest_xml = out_dir / f"{name}.xml"
        if not dest_xml.exists():
            dest_xml.write_text(_XML_TEMPLATE.format(name=name))

        print(f"  Object assets: {name}/")


# ---------------------------------------------------------------------------
# Sequence loading & processing
# ---------------------------------------------------------------------------

def load_omomo_seq_file(seq_file_path: str) -> list[dict]:
    """
    Load an OMOMO sequence .p file (joblib pickle).

    The file contains a dict[int, dict] where each entry is one sequence with:
        - seq_name: str
        - trans: (T, 3) float32 — root translation
        - root_orient: (T, 3) float32 — root orientation in axis-angle
        - pose_body: (T, 63) float64 — 21 body joints in axis-angle
        - betas: (1, 16) float32 — shape parameters
        - gender: str
        - (plus object fields)

    Returns list of sequence dicts.
    """
    data = joblib.load(seq_file_path)
    return [data[k] for k in sorted(data.keys())]


def extract_object_poses(seq: dict) -> np.ndarray | None:
    """
    Extract object poses from an OMOMO sequence dict and convert to pipeline format.

    The pipeline expects (T, 7) with [qw, qx, qy, qz, x, y, z].
    OMOMO stores obj_com_pos (T, 3) — object center-of-mass in world Z-up frame —
    and obj_rot (T, 3, 3).

    obj_com_pos is used (not obj_trans) because it represents the geometric center
    of the object matching MuJoCo's body origin convention, and is already in Z-up
    world frame. This matches how InterAct/InterMimic process OMOMO data.

    Returns None if object data is absent.
    """
    if "obj_com_pos" not in seq or "obj_rot" not in seq:
        return None

    obj_trans = seq["obj_com_pos"]   # T X 3, Z-up world frame (center of mass)
    obj_rot = seq["obj_rot"]         # T X 3 X 3

    # Rotation matrix → quaternion [qx, qy, qz, qw] → reorder to [qw, qx, qy, qz]
    quat_xyzw = Rotation.from_matrix(obj_rot).as_quat()  # T X 4 (xyzw)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]               # T X 4 (wxyz)

    return np.concatenate([quat_wxyz, obj_trans], axis=1).astype(np.float32)  # T X 7


def process_sequence(seq: dict, bm_dict: dict, num_body_joints: int = 22) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Run SMPLH FK on one sequence and return global joint positions + height + object poses.

    Args:
        seq: Sequence dict from load_omomo_seq_file.
        bm_dict: Dict of BodyModel instances (male/female/neutral).
        num_body_joints: Number of joints to keep (22 = body joints, no hands).

    Returns:
        global_joint_positions: (T, num_body_joints, 3)
        height: float — subject height in metres
        object_poses: (T, 7) [qw, qx, qy, qz, x, y, z] or None
    """
    T = seq["trans"].shape[0]
    trans = torch.from_numpy(seq["trans"]).float()
    root_orient = torch.from_numpy(seq["root_orient"]).float()
    pose_body = torch.from_numpy(seq["pose_body"].astype(np.float32))
    betas = torch.from_numpy(seq["betas"]).float().reshape(1, -1)
    gender = str(seq["gender"])
    if gender not in bm_dict:
        gender = "neutral"

    bm = bm_dict[gender]
    with torch.no_grad():
        out = bm(
            trans=trans,
            root_orient=root_orient,
            pose_body=pose_body,
            betas=betas.expand(T, -1),
        )

    global_joint_positions = out.Jtr[:, :num_body_joints, :].detach().cpu().numpy()  # T X 22 X 3
    height = compute_height(bm_dict, betas, gender)
    object_poses = extract_object_poses(seq)

    return global_joint_positions, height, object_poses


# ---------------------------------------------------------------------------
# Config & main
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Configuration for processing OMOMO data."""

    omomo_root_folder: str = "/path/to/omomo/data"
    """Root folder containing the OMOMO .p files and captured_objects/ subfolder."""

    output_folder: str = "holosoma_data/datasets/omomo"
    """Output folder for processed .npz sequence files."""

    smplh_root_folder: str = "holosoma_data/datasets/smplh_models"
    """Root folder containing SMPLH model files (male/model.npz, female/model.npz, neutral/model.npz)."""

    objects_output_folder: str = "holosoma_data/objects"
    """Output folder for generated object assets (.obj + .urdf + .xml)."""


def main(cfg: Config) -> None:
    os.makedirs(cfg.output_folder, exist_ok=True)

    # Generate object assets from captured_objects/
    captured_objects_dir = os.path.join(cfg.omomo_root_folder, "captured_objects")
    if os.path.isdir(captured_objects_dir):
        print(f"\nGenerating object assets from {captured_objects_dir} ...")
        generate_object_assets(captured_objects_dir, cfg.objects_output_folder)
    else:
        print(f"Warning: captured_objects/ not found in {cfg.omomo_root_folder}, skipping object assets.")

    # Process motion sequences
    bm_dict = prep_smplh_model(cfg.smplh_root_folder)

    for split in ("train", "test"):
        seq_file = os.path.join(cfg.omomo_root_folder, f"{split}_diffusion_manip_seq_joints24.p")
        if not os.path.exists(seq_file):
            print(f"Skipping {seq_file} (not found)")
            continue

        print(f"\nLoading {seq_file} ...")
        sequences = load_omomo_seq_file(seq_file)
        print(f"  {len(sequences)} sequences")

        for seq in sequences:
            seq_name: str = seq["seq_name"]
            output_path = os.path.join(cfg.output_folder, f"{seq_name}.npz")

            if os.path.exists(output_path):
                print(f"  Skipping {seq_name} (already exists)")
                continue

            global_joint_positions, height, object_poses = process_sequence(seq, bm_dict)

            save_data = dict(global_joint_positions=global_joint_positions, height=height)
            if object_poses is not None:
                save_data["object_poses"] = object_poses

            np.savez(output_path, **save_data)
            has_obj = object_poses is not None
            print(f"  Saved {seq_name} — shape={global_joint_positions.shape}, height={height:.3f}m, object={'yes' if has_obj else 'no'}")

    print("\nAll done.")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
