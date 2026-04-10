# Holosoma — TODO / Pipeline Observations

---

## 1. Stance foot constraint too restrictive on rotational motions

Motion: `SFU_0018_0018_DanceTurns001_stageii` (SFU / AMASS, SMPL-X)

```bash
source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && conda activate hsretargeting
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path holosoma_data/datasets/sfu/ \
    --task-type robot_only \
    --task-name SFU_0018_0018_DanceTurns001_stageii \
    --data_format smplx \
    --task-config.ground-range -10 10 \
    --save_dir demo_results/g1/robot_only/amass_smplx \
    --retargeter.debug \
    --retargeter.visualize
```

Problem: The robot needs to turn on itself with sometimes 1, sometimes 2 feet on the ground. The stance foot constraint locks a foot in both rotation and translation, which freezes the entire leg and makes the motion collapse into nonsense.

Lead: Relax the stance foot constraint to allow rotational/acrobatic motions. The risk is introducing foot skating and ground penetration. To compensate, implement a **DynaRetarget**-style post-processing layer (or concurrent layer) that specifically corrects foot skating / ground penetration without blocking acrobatic movements.

## 2. Pipeline Refactoring: Centralized Data Management (holosoma_data)

Goal: Create a 4th source module named holosoma_data to centralize all cross-cutting assets shared between the other three source directories.

Content of holosoma_data:
- Shared Assets: URDF and XML files for robots, objects, and environments/terrains.
- Source Datasets: Raw files used for retargeting (e.g., OMOMO, SFU, climb).
- Pipeline Outputs: Intermediate and final results from each stage, including retargeted motions and converted data formats.

Benefit: This refactoring will eliminate redundancy and provide a single source of truth for all data-heavy files used throughout the Holosoma ecosystem.

## 3. Upstream Synchronization

Task: Merge the latest improvements and bug fixes from the main holosoma repository (upstream) into the current working branch.

## GOAL :
I need to make the retargeting pipeline modular, meaning I should be able to call different calculation methods. After that, I will implement GMR to verify the baseline. Then, I’ll solve GMR while incorporating OmniRetarget constraints. Next, I will add the contact distance to the objective function (minimized function). I'll also account for the object frame's position and orientation. Following this, I will address the foot-skating constraints. Finally, I’ll solve the entire problem using LOIK.