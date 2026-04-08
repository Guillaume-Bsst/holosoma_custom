# TODO — `holosoma_data` : centralize all shared assets

## Context

Currently, data-heavy files are scattered across three source modules with duplication
and hard-coded cross-module paths:

| What | Current location | Used by |
|------|-----------------|---------|
| Robot URDFs + meshes | `holosoma_retargeting/models/g1/`, `holosoma/data/robots/g1/` | retargeting, training, inference |
| Object URDFs | `holosoma_retargeting/models/largebox/`, `holosoma/data/motions/.../` | retargeting, training |
| Raw source datasets | `holosoma_retargeting/demo_data/OMOMO_new/`, `demo_data/SFU/`, `demo_data/climb/` | retargeting only |
| Retargeted outputs | `holosoma/data/pipeline/g1_29dof/gmr/`, `.../omniretarget/` | training (consumed via `resolve_data_file_path`) |
| Converted motions | `holosoma_retargeting/converted_res/` | training |
| Terrain XMLs | `holosoma_retargeting/demo_data/climb/*/` | retargeting, training |

Notable hard-coded cross-module path:
```python
# pipeline/run.py line 50
_PIPELINE_DATA_DIR = Path(__file__).resolve().parents[3] / "holosoma" / "holosoma" / "data" / "pipeline"
```

---

## Goal

Create `src/holosoma_data/` as a 4th source module — a passive data store with no
Python logic — that becomes the **single source of truth** for all shared assets.

```
src/
├── holosoma/               # training + simulation
├── holosoma_inference/     # deployment
├── holosoma_retargeting/   # retargeting pipeline
└── holosoma_data/          # NEW — shared assets (no code)
    ├── robots/             # URDFs + meshes (g1, t1, h1, ...)
    ├── objects/            # object URDFs + meshes (largebox, ...)
    ├── environments/       # terrain XMLs and multi-box assets
    ├── datasets/           # raw source motion files (OMOMO, SFU, climb, ...)
    └── pipeline/           # pipeline outputs (retargeted, converted)
        ├── retargeted/     # output of holosoma_retargeting
        └── converted/      # output of data conversion step
```

---

## Steps

### Step 1 — Create `holosoma_data/` scaffold

- Create `src/holosoma_data/` with the folder structure above.
- Add a minimal `pyproject.toml` so it can be installed as a package if needed
  (makes `importlib.resources` resolution work across modules).
- Add a `holosoma_data/__init__.py` that exposes a `HOLOSOMA_DATA_ROOT: Path`
  constant pointing to the package root.

**Done when:** `from holosoma_data import HOLOSOMA_DATA_ROOT` works.

---

### Step 2 — Migrate robot and object assets

Move (not copy) the canonical URDF + mesh files:

| From | To |
|------|----|
| `holosoma_retargeting/models/g1/` | `holosoma_data/robots/g1/` |
| `holosoma_retargeting/models/t1/` | `holosoma_data/robots/t1/` |
| `holosoma_retargeting/models/h1/` | `holosoma_data/robots/h1/` |
| `holosoma_retargeting/models/largebox/` | `holosoma_data/objects/largebox/` |
| `holosoma/data/robots/g1/` | `holosoma_data/robots/g1/` (merge, resolve duplicates) |
| `holosoma/data/robots/t1/` | `holosoma_data/robots/t1/` |

Update all references:
- `holosoma_retargeting`: `models/g1/g1_29dof.urdf` → resolved via `HOLOSOMA_DATA_ROOT`
- `holosoma`: `holosoma/data/robots/...` → resolved via `HOLOSOMA_DATA_ROOT`
- `holosoma_inference`: update any URDF path configs

**Done when:** retargeting, training, and inference all load URDFs from `holosoma_data/`.

---

### Step 3 — Migrate source datasets

Move raw motion files out of `holosoma_retargeting/`:

| From | To |
|------|----|
| `holosoma_retargeting/demo_data/OMOMO_new/` | `holosoma_data/datasets/OMOMO_new/` |
| `holosoma_retargeting/demo_data/SFU/` | `holosoma_data/datasets/SFU/` |
| `holosoma_retargeting/demo_data/climb/` | `holosoma_data/datasets/climb/` |
| `holosoma_retargeting/demo_data/height_dict.pkl` | `holosoma_data/datasets/` |
| `holosoma_retargeting/data_utils/SFU/` | `holosoma_data/datasets/SFU_raw/` |

Update `--data_path` defaults and documentation examples in:
- `retargeters/README.md`
- `deploy/RETARGETING.md`
- `deploy/TRAINING.md`

**Done when:** retargeting commands work pointing to `holosoma_data/datasets/`.

---

### Step 4 — Redirect pipeline outputs

Change the pipeline output root from `holosoma/data/pipeline/` to
`holosoma_data/pipeline/retargeted/` and the conversion output from
`holosoma_retargeting/converted_res/` to `holosoma_data/pipeline/converted/`.

Files to update:
- `pipeline/run.py` line 50: `_PIPELINE_DATA_DIR`
- `data_conversion/convert_data_format_mj.py`: default output path
- `holosoma` training: `resolve_data_file_path` prefix `holosoma/data/pipeline/`
  → `holosoma_data/pipeline/retargeted/`

**Done when:** a full retargeting → conversion → training run reads and writes
exclusively to `holosoma_data/pipeline/`.

---

### Step 5 — Update `.gitignore` and CI

- Add `holosoma_data/datasets/` and `holosoma_data/pipeline/` to `.gitignore`
  (large binary data, not tracked).
- Keep `holosoma_data/robots/` and `holosoma_data/objects/` tracked (small,
  versioned assets).
- Update `holosoma_retargeting/.gitignore` to remove the now-migrated entries.
- Document in `deploy/README.md` how to populate `holosoma_data/datasets/`
  (download links for OMOMO, SFU, etc.).

---

## Migration checklist (per module)

- [ ] `holosoma_retargeting` — update `models/` references, `demo_data/` paths, `_PIPELINE_DATA_DIR`
- [ ] `holosoma` — update `holosoma/data/robots/`, `data/pipeline/` references, `resolve_data_file_path` prefix
- [ ] `holosoma_inference` — update any URDF path in configs
- [ ] `deploy/RETARGETING.md` — update all `--data_path` and `--robot-urdf` examples
- [ ] `deploy/TRAINING.md` — update motion file paths
- [ ] `.gitignore` (root + per-module) — update ignored paths
