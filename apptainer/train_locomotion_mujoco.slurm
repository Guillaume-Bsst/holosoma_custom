#!/bin/bash
#SBATCH --job-name=g1_fastsac_train
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p willow
#SBATCH -A willow

mkdir -p slurm_logs
mkdir -p $SCRATCH/holosoma_logs
mkdir -p $SCRATCH/wandb_cache

export APPTAINER_CACHEDIR=$SCRATCH/.apptainer_cache
export APPTAINER_TMPDIR=$TMP_DIR/.apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    /home/gbesset/WILLOW/holosoma/apptainer/holosoma.sif bash -c "
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export MUJOCO_GL=egl && \
    cd /workspace/holosoma && \
    python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof \
        simulator:mjwarp \
        logger:wandb \
        --logger.video.enabled=False
"

