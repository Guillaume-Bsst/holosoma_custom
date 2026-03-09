#!/bin/bash
set -e

export LANG=C.UTF-8
export DEBIAN_FRONTEND=noninteractive
export WORKSPACE_DIR="$(pwd)"
export CONDA_ROOT="$HOME/.holosoma_deps/miniconda3"
export PATH="$CONDA_ROOT/bin:$PATH"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMNI_KIT_ALLOW_ROOT=1

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    cmake build-essential swig curl wget unzip git \
    lsb-release ca-certificates \
    xvfb libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 \
    libxcursor1 libxdamage1 libglu1-mesa libegl1 libopengl0 libgl1 libosmesa6

if [ ! -d "$CONDA_ROOT" ]; then
    mkdir -p "$HOME/.holosoma_deps"
    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -u -p "$CONDA_ROOT"
    rm /tmp/miniconda.sh
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda config --set always_yes true
conda config --set channel_priority strict
conda config --set ssl_verify false

"$CONDA_ROOT/bin/pip" install uv

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

function pip() {
    env UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLATIONS=4 UV_NO_CACHE=1 uv pip "$@"
}
export -f pip

cd "$WORKSPACE_DIR/scripts"
chmod +x *.sh

for script in *.sh; do
    sed -i 's/\[unitree,booster\]/[unitree]/g' "$script"
    sed -i 's/\[unitree, booster\]/[unitree]/g' "$script"
done

export ACCEPT_EULA=Y
yes Yes | ./setup_isaacsim.sh
./setup_isaacgym.sh
./setup_mujoco.sh
./setup_inference.sh
./setup_retargeting.sh