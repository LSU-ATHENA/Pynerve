#!/usr/bin/env bash
# Pynerve GPU Self-Hosted Runner Setup
# Turns a fresh Ubuntu machine with an NVIDIA GPU into a GitHub Actions
# self-hosted runner for the LSU-ATHENA/Pynerve repository.
#
# Usage:
#   chmod +x tools/setup-gpu-runner.sh
#   ./tools/setup-gpu-runner.sh [OPTIONS]
#
# Options:
#   --labels LABELS       Extra runner labels (default: "gpu")
#   --cuda-version VER    CUDA version to install (default: 12.4)
#   --runner-version VER  Actions runner version (default: latest)
#   --ephemeral           Register as ephemeral runner (self-destructs after 1 job)
#   --dry-run             Print commands without executing
#   --help                Show this help
#
# Prerequisites:
#   - Ubuntu 22.04 or 24.04
#   - NVIDIA GPU with drivers installed (nvidia-smi must work)
#   - GitHub PAT or runner registration token
#   - sudo access
#
# After setup, register via:
#   cd actions-runner && ./config.sh --url https://github.com/LSU-ATHENA/Pynerve --token <TOKEN>

set -euo pipefail

# Defaults
LABELS="gpu"
CUDA_VERSION="12.4"
CUDA_VERSION_SHORT="${CUDA_VERSION//./-}"
RUNNER_VERSION="latest"
EPHEMERAL=""
DRY_RUN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --labels)
            LABELS="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --runner-version)
            RUNNER_VERSION="$2"
            shift 2
            ;;
        --ephemeral)
            EPHEMERAL="--ephemeral"
            shift
            ;;
        --dry-run)
            DRY_RUN="echo"
            shift
            ;;
        --help)
            sed -n '2,/^$/p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Pynerve GPU Runner Setup ==="
echo "Labels:      ${LABELS}"
echo "CUDA:        ${CUDA_VERSION}"
echo "Ephemeral:   ${EPHEMERAL:-false}"
echo ""

# 1. Verify NVIDIA GPU
echo "[1/6] Verifying NVIDIA GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first:"
    echo "  sudo apt-get install -y nvidia-driver-535"
    exit 1
fi
$DRY_RUN nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 2. Install CUDA toolkit (if needed)
echo "[2/6] Checking CUDA ${CUDA_VERSION}..."
if ! command -v nvcc &>/dev/null || ! nvcc --version | grep -q "release ${CUDA_VERSION}"; then
    echo "Installing CUDA ${CUDA_VERSION}..."
    $DRY_RUN wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    $DRY_RUN sudo dpkg -i cuda-keyring_1.1-1_all.deb
    $DRY_RUN sudo apt-get update
    $DRY_RUN sudo apt-get install -y "cuda-toolkit-${CUDA_VERSION_SHORT}"
    echo 'export CUDA_HOME=/usr/local/cuda' | $DRY_RUN sudo tee /etc/profile.d/cuda.sh
    echo 'export PATH="${CUDA_HOME}/bin:${PATH}"' | $DRY_RUN sudo tee -a /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"' | $DRY_RUN sudo tee -a /etc/profile.d/cuda.sh
fi
if [ -z "$DRY_RUN" ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
$DRY_RUN nvcc --version

# 3. Install system build dependencies
echo "[3/6] Installing build dependencies..."
$DRY_RUN sudo apt-get update
$DRY_RUN sudo apt-get install -y --no-install-recommends \
    build-essential g++-13 gcc-13 \
    ninja-build cmake pkg-config \
    libeigen3-dev libblas-dev liblapack-dev \
    libomp-dev libtbb-dev \
    ccache git curl wget ca-certificates \
    python3.11 python3.11-dev python3.11-venv python3-pip

$DRY_RUN sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100
$DRY_RUN sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

# 4. Install Python dependencies
echo "[4/6] Installing Python dependencies..."
$DRY_RUN python3.11 -m pip install --no-cache-dir -U pip
$DRY_RUN python3.11 -m pip install --no-cache-dir \
    torch --index-url "https://download.pytorch.org/whl/cu$(echo ${CUDA_VERSION} | tr -d '.')"
$DRY_RUN python3.11 -m pip install --no-cache-dir \
    ninja scikit-build-core pybind11 \
    numpy scipy \
    pytest pytest-xdist pytest-asyncio pytest-cov pytest-benchmark hypothesis \
    psutil scikit-learn numba networkx matplotlib build

# 5. Download and configure Actions Runner
echo "[5/6] Downloading GitHub Actions runner..."
RUNNER_DIR="$HOME/actions-runner"
if [ ! -d "$RUNNER_DIR" ]; then
    $DRY_RUN mkdir -p "$RUNNER_DIR"
    $DRY_RUN cd "$RUNNER_DIR"
    if [ "$RUNNER_VERSION" = "latest" ]; then
        RUNNER_URL=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | \
            grep "browser_download_url.*actions-runner-linux-x64" | cut -d '"' -f 4)
    else
        RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
    fi
    $DRY_RUN curl -o actions-runner-linux-x64.tar.gz -L "$RUNNER_URL"
    $DRY_RUN tar xzf actions-runner-linux-x64.tar.gz
    $DRY_RUN rm actions-runner-linux-x64.tar.gz
fi

# 6. Print registration instructions
echo ""
echo "=== [6/6] Runner downloaded — now register it ==="
echo ""
echo "1. Get a runner registration token:"
echo "   Go to: https://github.com/LSU-ATHENA/Pynerve/settings/actions/runners/new"
echo "   Select 'New self-hosted runner' → Linux → x64"
echo ""
echo "2. Register the runner:"
echo "   cd ${RUNNER_DIR}"
echo "   ./config.sh \\"
echo "     --url https://github.com/LSU-ATHENA/Pynerve \\"
echo "     --token <YOUR_TOKEN> \\"
echo "     --labels self-hosted,linux,x64,${LABELS} \\"
echo "     --name pynerve-gpu-$(hostname) \\"
echo "     ${EPHEMERAL}"
echo ""
echo "3. Install as a service (auto-start on boot):"
echo "   sudo ./svc.sh install"
echo "   sudo ./svc.sh start"
echo ""
echo "4. Verify the runner appears:"
echo "   https://github.com/LSU-ATHENA/Pynerve/settings/actions/runners"
echo ""
echo "=== Setup complete! ==="
