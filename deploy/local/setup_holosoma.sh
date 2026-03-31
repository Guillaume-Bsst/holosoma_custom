#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LANG=C.UTF-8
export DEBIAN_FRONTEND=noninteractive
export CONDA_ROOT="$HOME/.holosoma_deps/miniconda3"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMNI_KIT_ALLOW_ROOT=1

# ── 1. Install miniconda ───────────────────────────────────────────────────────

if [ ! -d "$CONDA_ROOT" ]; then
    mkdir -p "$HOME/.holosoma_deps"
    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -u -p "$CONDA_ROOT"
    rm /tmp/miniconda.sh
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate base

conda config --set always_yes true
conda config --set channel_priority strict
conda config --set ssl_verify false

# Remove Anaconda cloud plugins shipped with recent Miniconda — they pull pydantic
# as a dependency and break conda entry points in sub-environments.
conda remove -n base --force anaconda-auth conda-anaconda-tos 2>/dev/null || true

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# ── 2. Install base build tools via conda-forge (no sudo needed) ──────────────

# cmake, make, swig: required by several setup scripts
conda install -y -c conda-forge cmake make swig
# C/C++ compilers if not available on the system
if ! command -v gcc &>/dev/null; then
    conda install -y -c conda-forge compilers
fi
# xvfb-run: needed for headless rendering (IsaacSim / IsaacGym)
if ! command -v xvfb-run &>/dev/null; then
    conda install -y -c conda-forge xorg-xvfb 2>/dev/null \
        || echo "Warning: xvfb-run not found and could not be installed via conda-forge." \
               "IsaacSim/IsaacGym headless mode will not work without it."
fi

# ── 3. Fake lsb_release (some sub-scripts may check the OS codename) ─────────

mkdir -p "$HOME/.holosoma_deps/bin"
cat > "$HOME/.holosoma_deps/bin/lsb_release" << 'LSBEOF'
#!/bin/bash
case "$1" in
    -is|-i) echo "Ubuntu" ;;
    -rs|-r) echo "22.04" ;;
    -cs|-c) echo "jammy" ;;
    -a) echo -e "Distributor ID:\tUbuntu\nRelease:\t22.04\nCodename:\tjammy\nDescription:\tUbuntu 22.04 LTS" ;;
    *) echo "Ubuntu 22.04 jammy" ;;
esac
LSBEOF
chmod +x "$HOME/.holosoma_deps/bin/lsb_release"
export PATH="$HOME/.holosoma_deps/bin:$PATH"

export OS=linux
export OS_NAME=Linux
export DISTRIB_ID=Ubuntu
export DISTRIB_RELEASE=22.04

# ── 4. Speed up pip installs with uv ─────────────────────────────────────────

"$CONDA_ROOT/bin/python" -m pip install uv

export UV_BIN="$CONDA_ROOT/bin/uv"
if [ ! -x "$UV_BIN" ]; then
    UV_BIN="$(command -v uv 2>/dev/null || echo "")"
fi
function pip() {
    if [ -n "$UV_BIN" ] && [ -x "$UV_BIN" ]; then
        env UV_CONCURRENT_DOWNLOADS=8 UV_CONCURRENT_INSTALLATIONS=8 UV_NO_CACHE=1 "$UV_BIN" pip "$@"
    else
        python -m pip "$@"
    fi
}
export -f pip

# ── 5. Run individual setup scripts ──────────────────────────────────────────

export ACCEPT_EULA=Y

echo "==> setup_isaacsim.sh"
yes Yes | bash "$SCRIPT_DIR/setup_isaacsim.sh"

# Apply packaging patches required after IsaacLab install
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate hssim

echo "Applying patches for packaging, setuptools and flatdict..."
python -m pip install packaging==23.0
python -m pip install "setuptools<70.0.0"
python -m pip install flatdict==4.0.1 --no-build-isolation

# Re-run isaaclab install in case it needs the patched packages
ISAACLAB_DIR="$HOME/.holosoma_deps/IsaacLab"
if [ -f "$ISAACLAB_DIR/source/isaaclab/setup.py" ] || [ -f "$ISAACLAB_DIR/source/isaaclab/pyproject.toml" ]; then
    cd "$ISAACLAB_DIR/source/isaaclab"
    python -m pip install -e . || true
fi

conda deactivate

echo "==> setup_isaacgym.sh"
bash "$SCRIPT_DIR/setup_isaacgym.sh"

echo "==> setup_mujoco.sh"
bash "$SCRIPT_DIR/setup_mujoco.sh"

echo "==> setup_inference.sh"
bash "$SCRIPT_DIR/setup_inference.sh"

echo "==> setup_retargeting.sh"
bash "$SCRIPT_DIR/setup_retargeting.sh"

echo ""
echo "==========================================="
echo "Holosoma local setup complete."
echo "==========================================="