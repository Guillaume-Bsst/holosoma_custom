from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import torch
import trimesh
import tyro
from human_body_prior.body_model.body_model import BodyModel  # type: ignore[import-not-found]
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# SMPLH model
# ---------------------------------------------------------------------------

def prep_smplx_model(smplx_root_folder: str) -> dict:
    """Load SMPLX BodyModel instances for male, female and neutral genders."""
    num_betas = 16
    bm_dict = {}
    for gender in ("male", "female", "neutral"):
        bm_fname = os.path.join(smplx_root_folder, f"SMPLX_{gender.upper()}.npz")
        bm_dict[gender] = BodyModel(bm_fname=bm_fname, num_betas=num_betas)
    return bm_dict


def compute_height(bm_dict: dict, betas: torch.Tensor, gender: str) -> float:
    """Compute subject height from SMPLX T-pose vertices."""
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
        - obj_com_pos: (T, 3) — object center-of-mass position, Z-up world frame
        - obj_rot: (T, 3, 3) — object rotation matrix

    Returns list of sequence dicts.
    """
    data = joblib.load(seq_file_path)
    return [data[k] for k in sorted(data.keys())]


# ---------------------------------------------------------------------------
# Helpers ported from InterAct/process/process_omomo.py
# ---------------------------------------------------------------------------

def _quat_fk(lrot: np.ndarray, lpos: np.ndarray, parents: np.ndarray):
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(_quat_mul_vec(gr[parents[i]], lpos[..., i:i+1, :]) + gp[parents[i]])
        gr.append(_quat_mul(gr[parents[i]], lrot[..., i:i+1, :]))
    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)


def _quat_ik(grot: np.ndarray, gpos: np.ndarray, parents: np.ndarray):
    Q = np.concatenate([
        grot[..., :1, :],
        _quat_mul(_quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
    ], axis=-2)
    X = np.concatenate([
        gpos[..., :1, :],
        _quat_mul_vec(_quat_inv(grot[..., parents[1:], :]),
                      gpos[..., 1:, :] - gpos[..., parents[1:], :]),
    ], axis=-2)
    return Q, X


def _quat_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0,x1,x2,x3 = x[...,0:1],x[...,1:2],x[...,2:3],x[...,3:4]
    y0,y1,y2,y3 = y[...,0:1],y[...,1:2],y[...,2:3],y[...,3:4]
    return np.concatenate([
        y0*x0 - y1*x1 - y2*x2 - y3*x3,
        y0*x1 + y1*x0 - y2*x3 + y3*x2,
        y0*x2 + y1*x3 + y2*x0 - y3*x1,
        y0*x3 - y1*x2 + y2*x1 + y3*x0,
    ], axis=-1)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([1,-1,-1,-1], dtype=np.float32) * q


def _quat_mul_vec(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    t = 2.0 * np.cross(q[..., 1:], x)
    return x + q[..., 0:1] * t + np.cross(q[..., 1:], t)


def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + eps)


def _quat_between(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.sqrt(np.sum(x*x, axis=-1)*np.sum(y*y, axis=-1))[..., np.newaxis]
        + np.sum(x*y, axis=-1)[..., np.newaxis],
        np.cross(x, y),
    ], axis=-1)


def _rotate_at_frame_w_obj(X, Q, obj_x, obj_q, trans2joint, parents, n_past=1):
    """Canonicalise yaw at frame 0 (floor on z-plane). Ported from InterAct."""
    global_q, global_x = _quat_fk(Q, X, parents)
    key_glob_Q = global_q[:, n_past-1:n_past, 0:1, :]   # B,1,1,4
    forward = np.array([1,1,0]) * _quat_mul_vec(
        key_glob_Q, np.array([1,0,0])[np.newaxis,np.newaxis,np.newaxis,:]
    )
    forward = _normalize(forward)
    yrot = _normalize(_quat_between(np.array([1,0,0]), forward))
    new_glob_Q = _quat_mul(_quat_inv(yrot), global_q)
    new_glob_X = _quat_mul_vec(_quat_inv(yrot), global_x)
    new_obj_q  = _quat_mul(_quat_inv(yrot[:,0,:,:]), obj_q)
    obj_trans  = obj_x + trans2joint[:, np.newaxis, :]
    obj_trans  = _quat_mul_vec(_quat_inv(yrot[:,0,:,:]), obj_trans)
    obj_trans  = obj_trans - trans2joint[:, np.newaxis, :]
    Q, X = _quat_ik(new_glob_Q, new_glob_X, parents)
    return X, Q, obj_trans, new_obj_q


def _get_smplh_parents(smplh_npz_path: str) -> np.ndarray:
    data = np.load(smplh_npz_path)
    parents = data["kintree_table"][0, :22].copy()
    parents[0] = -1
    return parents


def process_sequence(
    seq: dict,
    bm_dict: dict,
    smplh_male_npz: str,
    obj_mesh_path: str | None = None,
    num_body_joints: int = 22,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Process one OMOMO sequence following the InterAct/InterMimic pipeline exactly:

    1. Build local joint positions (rest_offsets) and quaternions from raw poses.
    2. Canonicalise yaw (rotate_at_frame_w_obj) → canonical root trans + poses + obj_com_pos.
    3. Apply -π/2 around X on root orientation and translation (Y-up → Z-up).
    4. SMPLX FK with canonical+rotated poses → body vertices and pelvis.
    5. Reconstruct object world position: new_pelvis + rotated(obj_com - old_pelvis).
    6. Floor norm on Y axis (post -π/2 rotation): min(body_verts_y, obj_mesh_world_verts_y).
    7. Shift joints Y and obj_trans Y by -floor_y, which becomes Z in the final frame.

    Args:
        seq: Sequence dict from load_omomo_seq_file.
        bm_dict: Dict of SMPLX BodyModel instances (male/female/neutral).
        smplh_male_npz: Path to SMPLH male model.npz (for parent indices only).
        obj_mesh_path: Path to the unscaled object .obj mesh.
        num_body_joints: Number of joints to keep (22 = body without hands).

    Returns:
        global_joint_positions: (T, num_body_joints, 3)
        height: float — subject height in metres (from T-pose)
        object_poses: (T, 7) [qw, qx, qy, qz, x, y, z] or None
    """
    gender = str(seq["gender"])
    if gender not in bm_dict:
        gender = "neutral"
    bm = bm_dict[gender]

    trans       = seq["trans"].astype(np.float64)        # T X 3
    root_orient = seq["root_orient"].astype(np.float64)  # T X 3
    pose_body   = seq["pose_body"].astype(np.float64)    # T X 63
    rest_offsets = seq["rest_offsets"].astype(np.float64) # J X 3  (J=24)
    trans2joint  = seq["trans2joint"].astype(np.float64)  # 3
    betas        = torch.from_numpy(seq["betas"]).float().reshape(1, -1)
    T = trans.shape[0]
    J = rest_offsets.shape[0]

    has_object = "obj_com_pos" in seq and "obj_rot" in seq

    # ------------------------------------------------------------------
    # Step 1 — build local joint quats Q and positions X from rest_offsets
    # (ported directly from InterAct process_omomo.py)
    # ------------------------------------------------------------------
    body_pose_j = pose_body.reshape(T, 21, 3)
    extra       = np.zeros((T, J - 22, 3))
    joint_aa    = np.concatenate([root_orient[:, np.newaxis, :], body_pose_j, extra], axis=1)  # T X J X 3

    # axis-angle → rotation matrix → quaternion wxyz
    Q = Rotation.from_rotvec(joint_aa.reshape(-1, 3)).as_quat()   # (T*J) X 4 xyzw
    Q = Q[:, [3, 0, 1, 2]].reshape(T, J, 4)                       # T X J X 4 wxyz

    X = np.tile(rest_offsets[np.newaxis], (T, 1, 1))  # T X J X 3
    X[:, 0, :] = trans

    obj_rot_raw = seq["obj_rot"].astype(np.float64) if has_object else None  # T X 3 X 3
    obj_com_raw = seq["obj_com_pos"].astype(np.float64) if has_object else None  # T X 3

    if has_object:
        obj_q = Rotation.from_matrix(obj_rot_raw).as_quat()[:, [3,0,1,2]]  # T X 4 wxyz

    # ------------------------------------------------------------------
    # Step 2 — canonicalise yaw (rotate_at_frame_w_obj)
    # ------------------------------------------------------------------
    parents = _get_smplh_parents(smplh_male_npz)
    J_c = len(parents)  # 22

    if has_object:
        X_c, Q_c, new_obj_com, new_obj_q = _rotate_at_frame_w_obj(
            X[:, :J_c, :][np.newaxis], Q[:, :J_c, :][np.newaxis],
            obj_com_raw[np.newaxis], obj_q[np.newaxis],
            trans2joint[np.newaxis], parents,
        )
    else:
        X_c, Q_c, _, _ = _rotate_at_frame_w_obj(
            X[:, :J_c, :][np.newaxis], Q[:, :J_c, :][np.newaxis],
            np.zeros((1, T, 3)), np.tile([1,0,0,0], (1,T,1)).astype(np.float64),
            trans2joint[np.newaxis], parents,
        )

    # X_c, Q_c: (1, T, J_c, 3/4)  |  new_obj_com: (1, T, 3)
    new_trans       = X_c[0, :, 0, :]       # T X 3
    new_root_orient = Rotation.from_quat(Q_c[0, :, 0, :][:, [1,2,3,0]]).as_rotvec()  # T X 3
    new_pose_body   = Rotation.from_quat(
        Q_c[0, :, 1:22, :].reshape(-1, 4)[:, [1,2,3,0]]
    ).as_rotvec().reshape(T, 63)  # T X 63

    if has_object:
        obj_com_canon = new_obj_com[0]   # T X 3
        obj_rot_canon = Rotation.from_quat(new_obj_q[0][:, [1,2,3,0]]).as_matrix()  # T X 3 X 3

    # ------------------------------------------------------------------
    # Step 3 — apply -π/2 around X (InterAct process() step)
    # ------------------------------------------------------------------
    rx = Rotation.from_euler("x", -np.pi / 2)

    new_root_orient_rx = (rx * Rotation.from_rotvec(new_root_orient)).as_rotvec()
    new_trans_rx       = rx.apply(new_trans)

    # ------------------------------------------------------------------
    # Step 4 — SMPLX FK with canonical+rotated poses (first pass: pelvis before rx)
    # ------------------------------------------------------------------
    with torch.no_grad():
        out_pre = bm(
            trans=torch.from_numpy(new_trans).float(),
            root_orient=torch.from_numpy(new_root_orient).float(),
            pose_body=torch.from_numpy(new_pose_body).float(),
            betas=betas.expand(T, -1),
        )
    pelvis_pre = out_pre.Jtr[:, 0, :].detach().cpu().numpy()  # T X 3

    # Second pass: after -π/2
    with torch.no_grad():
        out = bm(
            trans=torch.from_numpy(new_trans_rx).float(),
            root_orient=torch.from_numpy(new_root_orient_rx).float(),
            pose_body=torch.from_numpy(new_pose_body).float(),
            betas=betas.expand(T, -1),
        )
    verts           = out.v.detach().cpu().numpy()                             # T X V X 3
    joint_positions = out.Jtr[:, :num_body_joints, :].detach().cpu().numpy()  # T X 22 X 3
    pelvis_post     = out.Jtr[:, 0, :].detach().cpu().numpy()                 # T X 3

    # ------------------------------------------------------------------
    # Step 5 — reconstruct object world position (InterAct process() step)
    # ------------------------------------------------------------------
    if has_object:
        obj_trans_delta = rx.apply(obj_com_canon - pelvis_pre)   # T X 3
        obj_world_trans = pelvis_post + obj_trans_delta           # T X 3
        obj_rot_rx      = (rx * Rotation.from_matrix(obj_rot_canon)).as_matrix()  # T X 3 X 3

    # ------------------------------------------------------------------
    # Step 6 — floor norm on Y axis (InterAct process() uses verts[:,y] after -π/2)
    # ------------------------------------------------------------------
    floor_y = verts[:, :, 1].min()

    if has_object and obj_mesh_path is not None:
        try:
            mesh        = trimesh.load(obj_mesh_path, force="mesh")
            local_verts = np.array(mesh.vertices)
            if "obj_scale" in seq:
                local_verts = local_verts * float(np.asarray(seq["obj_scale"]).flat[0])
            # World verts after rx rotation applied to obj_rot
            world_obj_v = (
                obj_rot_rx @ local_verts.T
            ).transpose(0, 2, 1) + obj_world_trans[:, np.newaxis, :]
            floor_y = min(floor_y, world_obj_v[:, :, 1].min())
        except Exception:
            pass

    # Shift Y by -floor_y
    joint_positions[:, :, 1] -= floor_y
    if has_object:
        obj_world_trans[:, 1] -= floor_y

    height = compute_height(bm_dict, betas, gender)

    if not has_object:
        return joint_positions, height, None

    quat_xyzw = Rotation.from_matrix(obj_rot_rx).as_quat()   # T X 4 xyzw
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]                   # T X 4 wxyz
    object_poses = np.concatenate(
        [quat_wxyz, obj_world_trans], axis=1
    ).astype(np.float32)  # T X 7  [qw, qx, qy, qz, x, y, z]

    return joint_positions, height, object_poses


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

    smplx_root_folder: str = "holosoma_data/datasets/smplx_models/models/smplx"
    """Root folder containing SMPLX model files (SMPLX_MALE.npz, SMPLX_FEMALE.npz, SMPLX_NEUTRAL.npz)."""

    smplh_male_npz: str = "holosoma_data/datasets/smplh_models/male/model.npz"
    """Path to SMPLH male model.npz (used only to read joint parent indices)."""

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
    bm_dict = prep_smplx_model(cfg.smplx_root_folder)
    smplh_male_npz = cfg.smplh_male_npz

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

            # Resolve object mesh path (e.g. "sub3_largebox_003" → objects/largebox/largebox.obj)
            obj_name = seq_name.split("_")[1]
            obj_mesh_path = os.path.join(cfg.objects_output_folder, obj_name, f"{obj_name}.obj")
            if not os.path.exists(obj_mesh_path):
                obj_mesh_path = None

            global_joint_positions, height, object_poses = process_sequence(
                seq, bm_dict, obj_mesh_path=obj_mesh_path
            )

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
