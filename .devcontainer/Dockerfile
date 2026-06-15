# Nerve Development Container
FROM nvidia/cuda:12.4-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libgtest-dev \
    lcov \
    gcovr \
    clang-format \
    clang-tidy \
    cppcheck \
    valgrind \
    gdb \
    lldb \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (single install, no venv duplication)
RUN python3.11 -m pip install --upgrade pip && pip install \
    torch \
    numpy \
    scipy \
    pytest \
    coverage \
    hypothesis \
    ruff \
    mypy \
    pre-commit \
    commitizen \
    bump-my-version \
    requests \
    jupyter \
    notebook \
    matplotlib \
    seaborn \
    plotly

# Create workspace directory
WORKDIR /workspace

# Copy setup script
COPY .devcontainer/setup.sh /tmp/setup.sh
RUN chmod +x /tmp/setup.sh

# Set up user (non-root for better permissions)
RUN useradd -m -s /bin/bash vscode
USER vscode

# Create build directory
RUN mkdir -p /workspace/build

# Pre-commit will be installed by postCreateCommand in devcontainer.json
# (not here, because .git is not available at build time)

# Set default command
CMD ["/bin/bash"]
