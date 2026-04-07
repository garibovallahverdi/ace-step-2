#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# ACE-Step 1.5 model selection
export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-xl-turbo}"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-1.7B}"
export ACESTEP_DEVICE="${ACESTEP_DEVICE:-auto}"
export ACESTEP_LM_BACKEND="${ACESTEP_LM_BACKEND:-vllm}"
export ACESTEP_INIT_LLM="${ACESTEP_INIT_LLM:-auto}"
export ACESTEP_DOWNLOAD_SOURCE="${ACESTEP_DOWNLOAD_SOURCE:-huggingface}"

# Optional: keep server startup fast by lazy-loading models (default behavior)
export ACESTEP_NO_INIT="${ACESTEP_NO_INIT:-true}"

exec uv run --no-sync acestep-api --host "$HOST" --port "$PORT"
