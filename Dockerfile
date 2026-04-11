FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive \
    ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo \
    ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B \
    ACESTEP_DEVICE=auto \
    ACESTEP_LM_BACKEND=vllm \
    ACESTEP_INIT_LLM=auto \
    ACESTEP_DOWNLOAD_SOURCE=huggingface \
    ACESTEP_NO_INIT=true

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

RUN python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --no-cache-dir --upgrade pip

# Install uv package manager
RUN python3.11 -m pip install --no-cache-dir uv
ENV PATH="/usr/local/bin:$PATH" \
    UV_PYTHON=python3.11

# Create a non-root user to run the application
RUN useradd -m -u 1001 appuser

# Set working directory
WORKDIR /app

# Copy local source code into image
COPY . /app

# Install project dependencies (ACE-Step 1.5 uses uv/pyproject)
RUN uv sync --no-dev

# Ensure target directories for volumes exist and have correct initial ownership
RUN mkdir -p /app/outputs /app/checkpoints /app/logs && \
    chown -R appuser:appuser /app/outputs /app/checkpoints /app/logs

# Prepare runtime files
RUN chmod +x /app/start.sh && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API and GUI ports
EXPOSE 8000 7865

VOLUME [ "/app/checkpoints", "/app/outputs", "/app/logs" ]

# Command to run the application (RunPod Serverless handler)
CMD ["python", "-m", "serverless.handler"]
