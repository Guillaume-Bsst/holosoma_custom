# TODO — Test Retargeter (standalone library)

## Architecture goal

`TestRetargeter` must become a **standalone library** living at `../Test/` alongside
`holosoma/` and `GMR/`, in the same way GMR is an independent package that holosoma
wraps. For now it is a copy of GMR's logic; the decoupling makes it possible to
iterate on the IK solver independently of the holosoma pipeline.

```
Documents/
├── holosoma/          ← pipeline (uses Test and GMR as plugins)
├── GMR/               ← external IK retargeter (read-only reference)
└── Test/              ← new standalone retargeter library (our own GMR clone)
```

---

## Step 0 — Create the `Test/` standalone package

**What:**
- Bootstrap `../Test/` as a pip-installable Python package mirroring the GMR
  package structure.
- Copy GMR's core files as a starting point:
  - `general_motion_retargeting/motion_retarget.py` → `test_retargeter/motion_retarget.py`
  - `general_motion_retargeting/ik_configs/smplx_to_g1.json` → `test_retargeter/ik_configs/`
  - `general_motion_retargeting/utils/smpl.py` → `test_retargeter/utils/smpl.py`
  - `general_motion_retargeting/params.py` → `test_retargeter/params.py`
- Expose a `TestMotionRetargeting` class with the same `retarget(human_data)` API
  as `GeneralMotionRetargeting`.
- Add a `pyproject.toml` so it installs with `pip install -e ../Test`.

**Done when:** `from test_retargeter import TestMotionRetargeting` works after
`pip install -e ../Test`.

---

## Step 1 — Wire `Test/` into the holosoma pipeline

**Files:** `retargeters/test.py`, `retargeters/gmr.py`

**What:**
- In `retargeters/test.py`, replace the current mink-based stub with the same
  auto-install + import pattern used for GMR (`retargeters/gmr.py`):
  - Clone `../Test/` if missing (or just `pip install -e ../Test`).
  - Import `TestMotionRetargeting` and use it identically to how `GMRRetargeter`
    uses `GeneralMotionRetargeting`.
- `TestRetargeter.retarget_motion()` calls `_load_gmr_frames()` (same AMASS loader)
  and iterates frame by frame, exactly like `GMRRetargeter`.
- The `test` method should produce **identical output** to `gmr` on the same input
  as a baseline sanity check.

**Done when:**
```bash
python examples/robot_retarget.py --retargeter-method test \
    --data_path data_utils/SFU/0005 \
    --task-name 0005_2FeetJump001_stageii \
    --data_format smplx --gmr.src_human smplx
```
runs without error and the result is numerically close to GMR.

---

## Step 2 — Validate GMR parity

Run both methods on the same sequence and compare qpos frame by frame:

```bash
# GMR
python examples/robot_retarget.py --retargeter-method gmr \
    --data_path data_utils/SFU/0005 --task-name 0005_2FeetJump001_stageii \
    --data_format smplx --gmr.src_human smplx

# Test
python examples/robot_retarget.py --retargeter-method test \
    --data_path data_utils/SFU/0005 --task-name 0005_2FeetJump001_stageii \
    --data_format smplx

# Compare
python -c "
import numpy as np
gmr  = np.load('path/to/gmr/retargeted.npz')['qpos']
test = np.load('path/to/test/retargeted.npz')['qpos']
print('MAE dof_pos:', np.abs(gmr[:, 7:] - test[:, 7:]).mean(), 'rad')
"
```

**Target:** MAE < 0.01 rad on joint DOFs.

---

## Future improvements (post-parity)

Once parity is established, the `Test/` library is the place to experiment:

- **Custom IK passes** — modify the two-pass solver without touching GMR
- **Alternative solvers** — swap daqp for proxqp or quadprog per robot
- **SMPLH support** — add a loader that estimates rotations from positions
  (for OMOMO/InterMimic data that has no raw pose params)
- **Real-time mode** — streaming API for teleoperation
