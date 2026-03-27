#!/bin/bash
# Exit on error, and print commands
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

# MuJoCo Warp version to install -- the repo is missing version tags and branches
# Arbitrarily chosen from mainline at the time we've ~tested against
MUJOCO_WARP_COMMIT="09ec1da"

# Parse command-line arguments
INSTALL_WARP=true  # Default: install warp (GPU-accelerated)

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-warp)
      INSTALL_WARP=false
      echo "MuJoCo Warp (GPU) installation disabled - CPU-only mode"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--no-warp]"
      echo ""
      echo "Options:"
      echo "  --no-warp      Skip MuJoCo Warp installation (CPU-only)"
      echo "  --help, -h     Show this help message"
      echo ""
      echo "Default: GPU-accelerated installation (WarpBackend + ClassicBackend)"
      echo ""
      echo "Examples:"
      echo "  # Initial setup (default: with GPU acceleration)"
      echo "  $0"
      echo ""
      echo "  # Setup without GPU acceleration (CPU-only)"
      echo "  $0 --no-warp"
      echo ""
      echo "Note: GPU acceleration requires NVIDIA driver >= 550.54.14"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--no-warp]"
      echo "Use --help for more information"
      exit 1
      ;;
  esac
done

# Use CONDA_ENV_NAME if provided, otherwise default to "hsmujoco"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsmujoco}
echo "conda environment name is set to: $CONDA_ENV_NAME"

source "${SCRIPT_DIR}/source_common.sh"
ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_$CONDA_ENV_NAME
WARP_SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_$CONDA_ENV_NAME_warp

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  OS_NAME="$(uname -s)"
  ARCH_NAME="$(uname -m)"

  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT

    if [[ "$OS_NAME" == "Linux" ]]; then
      MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OS_NAME" == "Darwin" ]]; then
      if [[ "$ARCH_NAME" == "arm64" ]]; then
        MINICONDA_INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
      else
        MINICONDA_INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
      fi
    else
      echo "Unsupported OS: $OS_NAME"
      exit 1
    fi

    curl "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" -o "$CONDA_ROOT/miniconda.sh"
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n $CONDA_ENV_NAME python=3.10 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate $CONDA_ENV_NAME

  # Install libstdcxx-ng to fix potential GLIBCXX issues (Linux only)
  if [[ "$OS_NAME" == "Linux" ]]; then
    conda install -c conda-forge -y libstdcxx-ng
  fi

  # Install ffmpeg for video encoding (consistent with other envs)
  conda install -c conda-forge -y ffmpeg

  # Install MuJoCo and related packages
  echo "Installing MuJoCo Python bindings..."
  pip install --upgrade pip

  pip install 'mujoco>=3.0.0'
  pip install mujoco-python-viewer

  # Install Holosoma packages
  echo "Installing Holosoma packages"
  pip install -U pip
  if [[ "$OS_NAME" == "Linux" ]]; then
    pip install -e "$ROOT_DIR/src/holosoma[unitree]"
  elif [[ "$OS_NAME" == "Darwin" ]]; then
    echo "Warning: only unitree support for osx"
    pip install -e "$ROOT_DIR/src/holosoma[unitree]"
  else
    echo "Unsupported OS: $OS_NAME"
    exit 1
  fi

  # Validate MuJoCo installation
  echo "Validating MuJoCo installation..."
  python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
  python -c "import mujoco_viewer; print('MuJoCo viewer imported successfully')"

  touch $SENTINEL_FILE
  echo ""
  echo "=========================================="
  echo "Base MuJoCo environment setup completed!"
  echo "=========================================="
  echo "Activate with: source scripts/source_mujoco_setup.sh"
  echo "=========================================="
fi

# Separate Warp installation (can be run independently after base install)
if [[ "$INSTALL_WARP" == "true" ]] && [[ ! -f $WARP_SENTINEL_FILE ]]; then
  echo ""
  echo "Installing MuJoCo Warp (GPU acceleration)..."

  source $CONDA_ROOT/bin/activate $CONDA_ENV_NAME

  # Check NVIDIA driver version (required for CUDA 12.4+)
  MIN_DRIVER_VERSION="550.54.14"
  DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)

  if [ -z "$DRIVER_VERSION" ] || [[ "$DRIVER_VERSION" < "$MIN_DRIVER_VERSION" ]]; then
    echo ""
    echo "ERROR: NVIDIA driver not found or too old!"
    if [ -z "$DRIVER_VERSION" ]; then
      echo "Status: No NVIDIA driver detected"
    else
      echo "Current driver:  $DRIVER_VERSION"
    fi
    echo "Minimum required: $MIN_DRIVER_VERSION (for CUDA 12.4+ support)"
    echo "After driver installation, re-run this script"
    echo "(or use --no-warp for CPU-only installation)"
    exit 1
  fi

  echo "NVIDIA driver version: $DRIVER_VERSION (meets minimum $MIN_DRIVER_VERSION)"

  if [[ ! -d $WORKSPACE_DIR/mujoco_warp ]]; then
    git clone https://github.com/google-deepmind/mujoco_warp.git $WORKSPACE_DIR/mujoco_warp && \
      git -C $WORKSPACE_DIR/mujoco_warp checkout ${MUJOCO_WARP_COMMIT}
  fi
  pip install uv
  uv pip install -e $WORKSPACE_DIR/mujoco_warp[dev,cuda]

  touch $WARP_SENTINEL_FILE

  echo ""
  echo "=========================================="
  echo "MuJoCo Warp installation completed!"
  echo "=========================================="
  echo "Activate with: source scripts/source_mujoco_setup.sh"
  echo "=========================================="
fi

echo ""
if [[ -f $WARP_SENTINEL_FILE ]]; then
  echo "MuJoCo environment ready with GPU acceleration (ClassicBackend + WarpBackend)"
elif [[ "$INSTALL_WARP" == "false" ]] && [[ -f $SENTINEL_FILE ]]; then
  echo "MuJoCo environment ready (CPU-only ClassicBackend)"
  echo ""
  echo "To add GPU acceleration later, run:"
  echo "  bash deploy/local/setup_mujoco.sh"
else
  echo "MuJoCo environment ready."
fi
echo "Use 'source scripts/source_mujoco_setup.sh' to activate."
