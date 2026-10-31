"""IsaacSim._object_cfg: the RigidObjectCfg of the object, built out of _setup_scene unchanged so that a subclass can
spawn otherwise (OTRNet: one object per env, IsaacLab's MultiAssetSpawnerCfg). isaacsim.py imports IsaacLab, which
only the Isaac env has: the move is checked on the source (ast), against the expression of 6fc32ec copied below.

    cd src/holosoma && $PYS -m pytest tests/simulator/test_isaacsim_object_cfg.py -q
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import holosoma

SRC = Path(holosoma.__file__).parent / "simulator/isaacsim/isaacsim.py"

# the object's RigidObjectCfg in IsaacSim._setup_scene at 6fc32ec, word for word
CFG_6FC32EC = '''RigidObjectCfg(
    prim_path=f"/World/envs/env_.*/Object",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=object_asset_urdf_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
    ),
)'''


def _tree():
    return ast.parse(SRC.read_text())


def _method(tree, name):
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "IsaacSim")
    return next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == name)


def test_object_cfg_builds_the_cfg_of_6fc32ec_from_the_resolved_path():
    fn = _method(_tree(), "_object_cfg")
    assert [a.arg for a in fn.args.args] == ["self", "urdf_path"]
    resolve, ret = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert ast.dump(resolve) == ast.dump(ast.parse("object_asset_urdf_path = resolve_data_file_path(urdf_path)").body[0])
    assert isinstance(ret, ast.Return) and ast.dump(ret.value) == ast.dump(ast.parse(CFG_6FC32EC).body[0].value)


def test_setup_scene_builds_the_object_through_it():
    fn = _method(_tree(), "_setup_scene")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_object_cfg"]
    assert len(calls) == 1
    assert ast.dump(calls[0].args[0]) == ast.dump(ast.parse("self.robot_config.object.object_urdf_path").body[0].value)
    built = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "RigidObjectCfg"
             and any(k.arg == "prim_path" and "Object" in ast.unparse(k.value) for k in n.keywords)]
    assert built == []


def test_setup_scene_reads_no_local_of_the_moved_block():
    tree = _tree()
    module_names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            module_names |= {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    fn = _method(tree, "_setup_scene")
    stored = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    stored |= {a.arg for n in ast.walk(fn) if isinstance(n, (ast.Lambda, ast.FunctionDef)) for a in n.args.args}
    read = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    free = read - stored - {a.arg for a in fn.args.args} - module_names - set(dir(builtins))
    assert not free, free
