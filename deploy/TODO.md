# Holosoma — TODO / Pipeline Observations

---

## 1. Stance foot constraint too restrictive on rotational motions

**Motion:** `SFU_0018_0018_DanceTurns001_stageii` (SFU / AMASS, SMPL-X)

```bash
source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && conda activate hsretargeting
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
    --data_path demo_data/SFU/ \
    --task-type robot_only \
    --task-name SFU_0018_0018_DanceTurns001_stageii \
    --data_format smplx \
    --task-config.ground-range -10 10 \
    --save_dir demo_results/g1/robot_only/amass_smplx \
    --retargeter.debug \
    --retargeter.visualize
```

**Problem:** The robot needs to turn on itself with sometimes 1, sometimes 2 feet on the ground. The stance foot constraint locks a foot in both rotation and translation, which freezes the entire leg and makes the motion collapse into nonsense.

**Lead:** Relax the stance foot constraint to allow rotational/acrobatic motions. The risk is introducing foot skating and ground penetration. To compensate, implement a **DynaRetarget**-style post-processing layer (or concurrent layer) that specifically corrects foot skating / ground penetration without blocking acrobatic movements.


g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)
Augmenter l'history