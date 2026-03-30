# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

echo "Setting up inference environment"

OS=$(uname -s)
ARCH=$(uname -m)

case $ARCH in
  "aarch64"|"arm64") ARCH="aarch64" ;;
  "x86_64") ARCH="x86_64" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

case $OS in
  "Linux")
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh"
    ;;
  "Darwin")
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

# Create overall workspace
source "${SCRIPT_DIR}/source_common.sh"
ENV_ROOT=$CONDA_ROOT/envs/hsinference

SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_inference

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install swig via conda if not already available (no sudo needed)
  command -v swig &>/dev/null || conda install -y -c conda-forge swig

  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl $MINICONDA_URL -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n hsinference python=3.11 \
      -c robostack-staging -c conda-forge --override-channels --channel-priority flexible \
      ros-humble-ros-base \
      ros-humble-rclpy \
      ros-humble-std-msgs \
      ros-humble-sensor-msgs \
      ros-humble-rmw-cyclonedds-cpp \
      ros-humble-rosidl-generator-dds-idl
  fi

  source $CONDA_ROOT/bin/activate hsinference

  export PYTHONNOUSERSITE=1
  unset PYTHONPATH

  # Install libstdcxx-ng to fix the error: `version `GLIBCXX_3.4.32' not found` on Ubuntu 24.04
  # Only needed on Linux (not macOS)
  if [[ $OS == "Linux" ]]; then
    conda install -c conda-forge -y libstdcxx-ng
  fi

  # Note: On macOS, only Unitree SDK is supported (Booster SDK is Linux-only)
  $ENV_ROOT/bin/python -m pip install -e $ROOT_DIR/src/holosoma_inference[unitree]

  # Setup a few things for ARM64 Linux (G1 Jetson)
  if [[ $OS == "Linux" && $ARCH == "aarch64" ]]; then
    # nvpmodel may not be available or may require sudo — ignore failures
    nvpmodel -m 0 2>/dev/null || true
    $ENV_ROOT/bin/python -m pip install "pin>=3.8.0"
  else
    if [[ ! -d $WORKSPACE_DIR/unitree_sdk2_python ]]; then
      git clone https://github.com/unitreerobotics/unitree_sdk2_python.git $WORKSPACE_DIR/unitree_sdk2_python
    fi
    $ENV_ROOT/bin/python -m pip install -e $WORKSPACE_DIR/unitree_sdk2_python/
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba install -y -n hsinference pinocchio -c conda-forge --override-channels
  fi

  cd $ROOT_DIR
  touch $SENTINEL_FILE
fi
