FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

ARG ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo
ARG ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ARG ACESTEP_DOWNLOAD_SOURCE=huggingface
ARG PRELOAD_MODELS=1
ARG HF_TOKEN

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/.huggingface \
    ACESTEP_API_HOST=0.0.0.0 \
    ACESTEP_API_PORT=8000 \
    ACESTEP_CONFIG_PATH=${ACESTEP_CONFIG_PATH} \
    ACESTEP_LM_MODEL_PATH=${ACESTEP_LM_MODEL_PATH} \
    ACESTEP_DEVICE=auto \
    ACESTEP_LM_BACKEND=vllm \
    ACESTEP_INIT_LLM=auto \
    ACESTEP_DOWNLOAD_SOURCE=${ACESTEP_DOWNLOAD_SOURCE} \
    ACESTEP_CHECKPOINTS_DIR=/app/checkpoints \
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

# Preload model weights into the image (optional, large)
RUN HF_TOKEN="$HF_TOKEN" PRELOAD_MODELS="$PRELOAD_MODELS" uv run --no-sync python - <<'PY'
import os
from acestep import model_downloader as md

if os.getenv("PRELOAD_MODELS", "1") != "1":
    raise SystemExit(0)

def _require(success: bool, message: str) -> None:
    if not success:
        raise SystemExit(message)

token = os.getenv("HF_TOKEN") or None
prefer = os.getenv("ACESTEP_DOWNLOAD_SOURCE") or None
config = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo").strip()
lm_model = os.getenv("ACESTEP_LM_MODEL_PATH", md.DEFAULT_LM_MODEL).strip()

checkpoints_dir = md.get_checkpoints_dir()

needs_main = config in md.SUBMODEL_REGISTRY or config == "acestep-v15-turbo"
if needs_main:
    ok, msg = md.download_main_model(
        checkpoints_dir=checkpoints_dir,
        token=token,
        prefer_source=prefer,
    )
    _require(ok, msg)

if config and config != "acestep-v15-turbo":
    if config not in md.SUBMODEL_REGISTRY:
        raise SystemExit(f"Unknown DiT model '{config}'.")
    ok, msg = md.download_submodel(
        model_name=config,
        checkpoints_dir=checkpoints_dir,
        token=token,
        prefer_source=prefer,
    )
    _require(ok, msg)

if lm_model and lm_model != md.DEFAULT_LM_MODEL:
    if lm_model not in md.SUBMODEL_REGISTRY:
        raise SystemExit(f"Unknown LM model '{lm_model}'.")
    ok, msg = md.download_submodel(
        model_name=lm_model,
        checkpoints_dir=checkpoints_dir,
        token=token,
        prefer_source=prefer,
    )
    _require(ok, msg)
PY

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

# Command to run the application (Verda serverless container entrypoint)
CMD ["uv", "run", "--no-sync", "python", "-m", "acestep.api_server"]
