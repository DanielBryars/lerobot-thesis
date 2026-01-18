# OpenPi (Pi0/Pi0.5) Training Docker Image with LeRobot v3.0 Support
# For use with vast.ai, RunPod, or similar GPU cloud services
#
# Docker Hub: aerdanielbryars101
# HuggingFace: danbhf
#
# Build: docker build -t aerdanielbryars101/openpi-training -f scripts/pi0/Dockerfile .
# Push:  docker push aerdanielbryars101/openpi-training
# Run:   docker run --gpus all -e HF_TOKEN=xxx -e WANDB_API_KEY=xxx aerdanielbryars101/openpi-training

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies + Python 3.11 from deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglew2.2 \
    libosmesa6 \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    patchelf \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libegl1 \
    libegl-dev \
    libgles2 \
    libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up Python 3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN python -m pip install --upgrade pip

# Set MuJoCo to use EGL for GPU-accelerated headless rendering
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Create working directory
WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Clone openpi
RUN git clone https://github.com/Physical-Intelligence/openpi.git /app/openpi

# ============================================
# PATCH OPENPI FOR LEROBOT v3.0 SUPPORT
# ============================================

# 1. Update lerobot dependency to use PyPI version >=0.4.0
RUN sed -i 's/"lerobot",/"lerobot>=0.4.0",/' /app/openpi/pyproject.toml

# 2. Remove numpy<2.0.0 constraints (lerobot 0.4.x requires numpy>=2)
RUN sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' /app/openpi/pyproject.toml && \
    sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' /app/openpi/packages/openpi-client/pyproject.toml

# 3. Remove the lerobot git pin from [tool.uv.sources]
RUN sed -i '/^lerobot = { git/d' /app/openpi/pyproject.toml

# 4. Remove the rlds dependency group (conflicts with numpy>=2, not needed for LeRobot)
RUN sed -i '/^rlds = \[/,/^\]/d' /app/openpi/pyproject.toml && \
    sed -i '/^dlimp = { git/d' /app/openpi/pyproject.toml

# 5. Fix lerobot import path (v0.4.x moved common.datasets to datasets)
RUN sed -i 's/lerobot.common.datasets/lerobot.datasets/g' /app/openpi/src/openpi/training/data_loader.py

# 6. Install openpi with patched dependencies using uv
WORKDIR /app/openpi
RUN uv sync

# Verify lerobot version
RUN .venv/bin/python -c "import lerobot; print(f'LeRobot version: {lerobot.__version__}')" && \
    .venv/bin/python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# ============================================
# COPY AND SETUP LEROBOT-THESIS
# ============================================

WORKDIR /app

# Copy our code
COPY . /app/lerobot-thesis

# Fix line endings (in case of Windows)
RUN find /app/lerobot-thesis -name "*.sh" -exec sed -i 's/\r$//' {} \; 2>/dev/null || true && \
    find /app/lerobot-thesis -name "*.py" -exec sed -i 's/\r$//' {} \; 2>/dev/null || true

# Apply SO-101 config patch to openpi
RUN python /app/lerobot-thesis/scripts/pi0/patch_openpi_so101.py --config-path=/app/openpi/src/openpi/training/config.py

# Set Python path
ENV PYTHONPATH=/app/openpi/src:/app/lerobot-thesis:/app/lerobot-thesis/scripts:$PYTHONPATH

# Create directories
RUN mkdir -p /app/outputs /app/datasets

# Set environment variables
ENV HF_HOME=/app/.cache/huggingface
ENV WANDB_DIR=/app/wandb

# Default working directory for training
WORKDIR /app/openpi

# Default command - show available configs
CMD ["uv", "run", "scripts/train.py", "--help"]
