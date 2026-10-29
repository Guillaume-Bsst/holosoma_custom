"""Resolve where/what free box to spawn in run_sim for object-carry sim-to-sim.

Two independent things are derived here so the spawned box always matches whatever object-carry
checkpoint is being played back, instead of relying on hand-tuned box_half_extent/box_mass/box_pos
config values that drift out of sync with the actual training data:

1. Geometry + mass: read directly from the object URDF (robot.object.object_urdf_path) the checkpoint
   was trained with -- the same file passed to training's --robot.object.object-urdf-path.
2. Spawn pose: read the box pose relative to the robot root at a given clip frame (the same clip
   passed to holosoma_inference's --task.object-motion-file) and re-anchor it to this scene's actual
   robot init_state pose, so the box lands where the policy expects it relative to the robot
   regardless of where the robot happens to stand in this particular MuJoCo scene.

Falls back to the static sim.box_pos/box_half_extent/box_mass config when the object URDF / motion
file aren't configured (e.g. plain decorative box, no object-carry checkpoint).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from holosoma.utils.path import resolve_data_file_path

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig
    from holosoma.config_types.simulator import SimEngineConfig


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _quat_mul_wxyz(_quat_mul_wxyz(q, qv), _quat_conj_wxyz(q))[1:]


def _parse_object_urdf_box(urdf_path: str) -> tuple[float, float]:
    """Derive (half_extent_m, mass_kg) from a single-link box object URDF + its mesh.

    Matches the objects_box32.urdf / objects_box36.urdf / objects_largebox.urdf shape used for
    object-carry training: one <link> with a box-shaped mesh centered at the origin and a
    <mass value=.../> tag. half_extent is measured directly off the mesh vertices rather than assumed,
    so it stays correct for any object of this shape without a config knob per box size.
    """
    resolved = resolve_data_file_path(urdf_path)
    root = ET.parse(resolved).getroot()

    mass_el = root.find(".//mass")
    if mass_el is None:
        raise ValueError(f"No <mass> tag found in object URDF: {resolved}")
    mass = float(mass_el.get("value"))

    mesh_el = root.find(".//collision/geometry/mesh")
    if mesh_el is None:
        mesh_el = root.find(".//visual/geometry/mesh")
    if mesh_el is None:
        raise ValueError(f"No mesh geometry found in object URDF: {resolved}")
    mesh_path = os.path.join(os.path.dirname(resolved), mesh_el.get("filename"))

    vertices = []
    with open(mesh_path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices found in object mesh: {mesh_path}")
    verts = np.asarray(vertices, dtype=np.float64)
    half_extent = float((verts.max(axis=0) - verts.min(axis=0)).max() / 2.0)

    return half_extent, mass


def _compute_box_spawn_pose(
    motion_file: str,
    timestep: int,
    robot_init_pos: tuple[float, float, float],
    robot_init_rot_xyzw: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (world_pos[3], world_quat_wxyz[4]) for the box at `timestep` of `motion_file`.

    Reads the clip's object pose relative to the robot root (the same relative transform
    holosoma_inference feeds the policy as obj_pos_b/obj_ori_b) and re-expresses it using the robot's
    ACTUAL init_state pose in this scene, so the box appears where the policy expects it relative to
    the robot even though the robot's spawn pose here is unrelated to the clip's own world frame.
    """
    data = np.load(resolve_data_file_path(motion_file))
    joint_pos = data["joint_pos"]
    obj_pos_w = data["object_pos_w"]
    obj_quat_w = data["object_quat_w"]  # wxyz

    t = int(np.clip(timestep, 0, joint_pos.shape[0] - 1))
    return _reanchor_yawxy(obj_pos_w[t], obj_quat_w[t], joint_pos[t, :7], robot_init_pos, robot_init_rot_xyzw)


def _yaw_of_wxyz(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _reanchor_yawxy(
    clip_pos: np.ndarray,
    clip_quat_wxyz: np.ndarray,
    clip_root: np.ndarray,
    robot_init_pos,
    robot_init_rot_xyzw,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a clip-world pose into this scene, anchoring by yaw + XY only.

    Rotates/translates about the clip's root-vs-robot-init delta restricted to yaw and the ground
    plane: heights stay EXACTLY as in the clip (both floors are at z=0), so nothing sinks into or
    hovers over the ground when the robot init z differs slightly from the clip's root z.
    """
    root_yaw = _yaw_of_wxyz(clip_root[3:7])
    robot_yaw = _yaw_of_wxyz(
        np.array([robot_init_rot_xyzw[3], robot_init_rot_xyzw[0], robot_init_rot_xyzw[1], robot_init_rot_xyzw[2]])
    )
    dyaw = robot_yaw - root_yaw
    c, s = np.cos(dyaw), np.sin(dyaw)

    rel_xy = np.asarray(clip_pos[:2], dtype=np.float64) - np.asarray(clip_root[:2], dtype=np.float64)
    world_xy = np.asarray(robot_init_pos[:2], dtype=np.float64) + np.array(
        [c * rel_xy[0] - s * rel_xy[1], s * rel_xy[0] + c * rel_xy[1]]
    )
    world_pos = np.array([world_xy[0], world_xy[1], float(clip_pos[2])])

    dyaw_quat = np.array([np.cos(dyaw / 2.0), 0.0, 0.0, np.sin(dyaw / 2.0)])
    world_quat = _quat_mul_wxyz(dyaw_quat, np.asarray(clip_quat_wxyz, dtype=np.float64))

    return world_pos, world_quat


def resolve_support_spawn(
    sim_cfg: SimEngineConfig, robot_config: RobotConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Resolve (vertices[N,3], faces[M,3], pos[3], quat_wxyz[4]) for the support table, or None.

    Implements SimEngineConfig.add_support: loads the REAL support mesh (clip-world coordinates,
    e.g. femto14_support_world.obj -- rotated plateau + legs, triangulated) and computes the rigid
    transform that re-anchors the whole clip world relative to the robot's init pose (same yaw+XY
    mapping as the free box, heights untouched). The caller attaches the mesh under a body at that
    transform, so the table keeps its exact shape and orientation -- an AABB box approximation is
    wrong here because the table is rotated ~-105 deg in the clip world.
    """
    if not (sim_cfg.add_support and sim_cfg.support_obj_file and sim_cfg.object_motion_file):
        return None

    mesh_path = resolve_data_file_path(sim_cfg.support_obj_file)
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(mesh_path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                # fan-triangulate just in case (trimesh exports are already triangles)
                faces.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))
    vertices = np.asarray(verts, dtype=np.float32)
    faces_arr = np.asarray(faces, dtype=np.int32)

    # Rigid transform mapping clip-world coordinates into this scene (yaw+XY anchoring):
    # world_pt = t + Rz(dyaw) @ clip_pt, derived from mapping the clip root onto the robot init.
    data = np.load(resolve_data_file_path(sim_cfg.object_motion_file))
    t_idx = int(np.clip(sim_cfg.object_motion_start_timestep, 0, data["joint_pos"].shape[0] - 1))
    origin, quat = _reanchor_yawxy(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        data["joint_pos"][t_idx, :7],
        robot_config.init_state.pos,
        robot_config.init_state.rot,
    )
    logger.info(
        f"Support table mesh '{sim_cfg.support_obj_file}': {len(vertices)} verts, "
        f"{len(faces_arr)} tris, scene transform pos={origin.round(3).tolist()}"
    )
    return vertices, faces_arr, origin, quat


def _mat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (w,x,y,z) quaternion (Shepperd's method)."""
    m00, m01, m02 = rot[0]
    m10, m11, m12 = rot[1]
    m20, m21, m22 = rot[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    return np.array([w, x, y, z])


def resolve_stairs_spawn(sim_cfg: SimEngineConfig, robot_config: RobotConfig):
    """Resolve a list of (pos[3], half_extents[3], quat_wxyz[4]) static boxes for a HoloV2
    staircase, one overlapping box per height LEVEL (landing + each tread) -- see
    `SceneManager.add_static_boxes` for why per-level boxes (not a single mesh) are needed for a
    non-convex shape like a staircase.

    Reads `stairs_params_file` -- the exact `stair_params.npz` the retargeting stage saved when
    it built the staircase mesh (riser/going/n_steps/s0/z_min/center_xy/forward/lateral/stances,
    plus `isaac_z_shift` applied to the Isaac-side clip's body_pos_w/joint_pos so the landing sits
    at z=0 there -- the SAME shift must be applied here for the two sims to agree). Each box's
    clip-world pose is re-anchored to THIS scene's robot init pose exactly like
    `resolve_support_spawn` does for the table (yaw+XY only, heights untouched).
    """
    if not (sim_cfg.add_stairs and sim_cfg.stairs_params_file and sim_cfg.object_motion_file):
        return None

    p = np.load(resolve_data_file_path(sim_cfg.stairs_params_file))
    riser, going, n_steps = float(p["riser"]), float(p["going"]), int(p["n_steps"])
    s0 = float(p["s0"])
    z_min = float(p["z_min"]) + float(p["isaac_z_shift"])
    center_xy = p["center_xy"].astype(np.float64)
    forward = p["forward"].astype(np.float64)
    lateral = p["lateral"].astype(np.float64)
    stances = p["stances"].astype(np.float64)

    xy0 = stances[:, :2] - center_xy[None, :]
    w_coord = xy0 @ lateral
    box_rot = np.stack(
        [np.array([forward[0], forward[1], 0.0]), np.array([lateral[0], lateral[1], 0.0]), np.array([0.0, 0.0, 1.0])],
        axis=1,
    )
    quat_clipworld = _mat_to_quat_wxyz(box_rot)

    data = np.load(resolve_data_file_path(sim_cfg.object_motion_file))
    t_idx = int(np.clip(sim_cfg.object_motion_start_timestep, 0, data["joint_pos"].shape[0] - 1))
    clip_root = data["joint_pos"][t_idx, :7]

    # Extend the landing (lowest level) and the far end to cover the FULL horizontal range the
    # clip's root actually visits, not just the "aesthetic" stair span derived from the ascending
    # footplants (stances). On Isaac the SDF softly extends past the literal mesh edge (a
    # continuous distance field), so a too-short mesh there is forgiving; a MuJoCo box geom is
    # not -- past its edge is a true void, which is what made the robot "climb into nothing"
    # before this fix.
    root_xy0 = data["joint_pos"][:, :2] - center_xy[None, :]
    root_s = root_xy0 @ forward
    root_w = root_xy0 @ lateral
    margin = 0.5
    s_lead0 = min(s0 - going, float(root_s.min()) - margin)
    s_end = max(s0 + (n_steps + 1) * going, float(root_s.max()) + margin)
    w_lo = min(float(w_coord.min()), float(root_w.min())) - 0.3
    w_hi = max(float(w_coord.max()), float(root_w.max())) + 0.3
    base_z = z_min - 0.5  # deep base so no gap shows under the lowest step at the scene border

    levels = [(s_lead0, z_min)] + [(s0 + n * going, z_min + (n + 1) * riser) for n in range(n_steps)]
    boxes = []
    for s_lead, z_top in levels:
        hx, hy, hz = (s_end - s_lead) / 2.0, (w_hi - w_lo) / 2.0, (z_top - base_z) / 2.0
        s_c, w_c, z_c = (s_lead + s_end) / 2.0, (w_lo + w_hi) / 2.0, (base_z + z_top) / 2.0
        pos_xy = center_xy + s_c * forward + w_c * lateral
        pos_clipworld = np.array([pos_xy[0], pos_xy[1], z_c])
        pos_world, quat_world = _reanchor_yawxy(
            pos_clipworld, quat_clipworld, clip_root, robot_config.init_state.pos, robot_config.init_state.rot
        )
        boxes.append((pos_world, np.array([hx, hy, hz]), quat_world))

    logger.info(
        f"Stairs '{sim_cfg.stairs_params_file}': {len(boxes)} convex boxes, "
        f"riser={riser:.3f} going={going:.3f} n_steps={n_steps} "
        f"s_lead0={s_lead0:.3f} s_end={s_end:.3f}"
    )
    logger.info(
        f"Stairs debug: clip_root (frame {t_idx}) xy={clip_root[:2].round(3).tolist()} -> "
        f"robot spawns at robot_config.init_state.pos={list(robot_config.init_state.pos)}. "
        f"Box world centers: " + ", ".join(f"L{i}={b[0].round(3).tolist()}" for i, b in enumerate(boxes))
    )
    return boxes


def robot_init_state_from_clip(sim_cfg: SimEngineConfig, robot_config: RobotConfig) -> RobotConfig | None:
    """Return a robot config whose init_state matches the clip's frame-0 root pose, or None.

    Implements SimEngineConfig.spawn_robot_at_clip_start: x, y and yaw come from
    object_motion_file's frame 0 (the same anchoring training's default-pose prepend uses);
    z, roll and pitch are kept from the configured init_state. Returns None when the flag or
    the motion file is absent.
    """
    if not (sim_cfg.spawn_robot_at_clip_start and sim_cfg.object_motion_file):
        return None

    import dataclasses  # noqa: PLC0415 -- only needed on this path

    data = np.load(resolve_data_file_path(sim_cfg.object_motion_file))
    root = np.asarray(data["joint_pos"], dtype=np.float64)[0, :7]  # [pos(3), quat wxyz(4)]
    w, x, y, z = root[3:7]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    init = robot_config.init_state
    new_pos = [float(root[0]), float(root[1]), init.pos[2]]
    # yaw-only rotation, xyzw (init roll/pitch are flat for humanoid spawn configs)
    new_rot = [0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))]
    logger.info(
        f"Robot init overridden from clip frame 0: pos={[round(v, 3) for v in new_pos]}, "
        f"yaw={np.degrees(yaw):.1f} deg (spawn_robot_at_clip_start)"
    )
    return dataclasses.replace(
        robot_config, init_state=dataclasses.replace(init, pos=new_pos, rot=new_rot)
    )


def resolve_box_spawn(
    sim_cfg: SimEngineConfig, robot_config: RobotConfig
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Resolve (pos[3], quat_wxyz[4], half_extent, mass) for the free box spawned by run_sim.

    See module docstring: prefers the object URDF / motion clip when configured, falls back to the
    static sim.box_* values otherwise.
    """
    urdf_path = robot_config.object.object_urdf_path
    if urdf_path:
        half_extent, mass = _parse_object_urdf_box(urdf_path)
        logger.info(
            f"Box geometry/mass derived from object URDF '{urdf_path}': "
            f"half_extent={half_extent:.3f} m, mass={mass:.3f} kg"
        )
    else:
        half_extent, mass = sim_cfg.box_half_extent, sim_cfg.box_mass

    if sim_cfg.object_motion_file:
        pos, quat = _compute_box_spawn_pose(
            sim_cfg.object_motion_file,
            sim_cfg.object_motion_start_timestep,
            robot_config.init_state.pos,
            robot_config.init_state.rot,
        )
        logger.info(
            f"Box spawn pose anchored from clip '{sim_cfg.object_motion_file}' "
            f"@t={sim_cfg.object_motion_start_timestep}: pos={pos.round(3).tolist()}"
        )
    else:
        pos = np.asarray(sim_cfg.box_pos, dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0])

    return pos, quat, half_extent, mass
