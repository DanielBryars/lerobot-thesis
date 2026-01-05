# LeRobot Training Docker Image
# For use with vast.ai, RunPod, or similar GPU cloud services
#
# Build: docker build -t lerobot-training .
# Run:   docker run --gpus all -e HF_TOKEN=xxx -e WANDB_API_KEY=xxx lerobot-training

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
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

# Set MuJoCo to use OSMesa for headless rendering
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Create working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY remote/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy the rest of the code
COPY . .

# Install the simulation plugin
RUN pip install -e src/lerobot_robot_sim --no-deps

# Install the local utils
RUN pip install -e . --no-deps 2>/dev/null || true

# Create directory for outputs
RUN mkdir -p /app/outputs /app/datasets

# Set environment variables for HuggingFace and WandB (to be overridden at runtime)
ENV HF_HOME=/app/.cache/huggingface
ENV WANDB_DIR=/app/wandb

# Entry point
ENTRYPOINT ["python", "remote/train_remote.py"]
