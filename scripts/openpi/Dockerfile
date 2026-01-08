# OpenPi (Pi0/Pi0.5) Training Docker Image
# For use with vast.ai, RunPod, or similar GPU cloud services
#
# Docker Hub: aerdanielbryars101
# HuggingFace: danbhf
#
# Build: docker build -t aerdanielbryars101/openpi-training -f scripts/openpi/Dockerfile .
# Push:  docker push aerdanielbryars101/openpi-training
# Run:   docker run --gpus all -e HF_TOKEN=xxx -e WANDB_API_KEY=xxx aerdanielbryars101/openpi-training

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
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

# Set up Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

# Set MuJoCo to use EGL for GPU-accelerated headless rendering
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Create working directory
WORKDIR /app

# Install JAX with CUDA support first
RUN pip install --no-cache-dir "jax[cuda12]"

# Install core dependencies
RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    pillow \
    tqdm \
    wandb \
    huggingface_hub \
    optax \
    flax \
    transformers

# Clone and install openpi
RUN git clone https://github.com/Physical-Intelligence/openpi.git /app/openpi && \
    cd /app/openpi && \
    GIT_LFS_SKIP_SMUDGE=1 pip install -e . || echo "openpi install may have warnings"

# Copy our code
COPY scripts/openpi /app/lerobot-thesis/scripts/openpi
COPY utils /app/lerobot-thesis/utils
COPY src /app/lerobot-thesis/src

# Set Python path (include scripts dir for openpi imports)
ENV PYTHONPATH=/app/lerobot-thesis:/app/lerobot-thesis/scripts:/app/openpi:$PYTHONPATH

# Create directories
RUN mkdir -p /app/outputs /app/datasets

# Set environment variables
ENV HF_HOME=/app/.cache/huggingface
ENV WANDB_DIR=/app/wandb

# Default command - show help
CMD ["python", "/app/lerobot-thesis/scripts/openpi/train_pi0.py", "--help"]
