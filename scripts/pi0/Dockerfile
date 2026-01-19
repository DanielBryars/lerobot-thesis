# LeRobot Pi0 Training Docker Image
# For vast.ai training on H100/A100 GPUs
#
# Build:
#   docker build -t aerdanielbryars101/lerobot-pi0:latest -f scripts/pi0/Dockerfile .
#
# Push:
#   docker push aerdanielbryars101/lerobot-pi0:latest

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies including ffmpeg with all libs for torchcodec
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set video backend to pyav (more compatible than torchcodec)
ENV VIDEO_BACKEND=pyav

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create app directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install LeRobot with Pi0 support
RUN uv pip install --system "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"

# Install additional dependencies
RUN uv pip install --system wandb huggingface_hub safetensors

# Create training script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Default values\n\
DATASET=${DATASET:-"danbhf/sim_pick_place_157ep"}\n\
STEPS=${STEPS:-5000}\n\
BATCH_SIZE=${BATCH_SIZE:-32}\n\
JOB_NAME=${JOB_NAME:-"pi0_training"}\n\
REPO_ID=${REPO_ID:-""}\n\
\n\
echo "================================================"\n\
echo "LeRobot Pi0 Training"\n\
echo "================================================"\n\
echo "Dataset: $DATASET"\n\
echo "Steps: $STEPS"\n\
echo "Batch size: $BATCH_SIZE"\n\
echo "Job name: $JOB_NAME"\n\
echo "Push to: $REPO_ID"\n\
echo "================================================"\n\
\n\
# Build command\n\
CMD="lerobot-train \\\n\
    --dataset.repo_id=$DATASET \\\n\
    --policy.type=pi0 \\\n\
    --policy.pretrained_path=lerobot/pi0_base \\\n\
    --output_dir=/app/outputs/$JOB_NAME \\\n\
    --job_name=$JOB_NAME \\\n\
    --policy.gradient_checkpointing=true \\\n\
    --policy.dtype=bfloat16 \\\n\
    --policy.device=cuda \\\n\
    --steps=$STEPS \\\n\
    --batch_size=$BATCH_SIZE \\\n\
    --wandb.enable=true"\n\
\n\
if [ -n "$REPO_ID" ]; then\n\
    CMD="$CMD --policy.repo_id=$REPO_ID"\n\
fi\n\
\n\
echo "Running: $CMD"\n\
eval $CMD\n\
' > /app/train.sh && chmod +x /app/train.sh

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
CMD ["/app/train.sh"]
