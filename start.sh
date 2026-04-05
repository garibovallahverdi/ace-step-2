#!/usr/bin/env bash
set -euo pipefail

APP_MODE="${APP_MODE:-api}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
DEVICE_ID="${DEVICE_ID:-0}"
BF16="${BF16:-true}"
TORCH_COMPILE="${TORCH_COMPILE:-false}"
CPU_OFFLOAD="${CPU_OFFLOAD:-false}"
OVERLAPPED_DECODE="${OVERLAPPED_DECODE:-false}"

if [ "$APP_MODE" = "gui" ]; then
  exec acestep \
    --server_name 0.0.0.0 \
    --port "${GUI_PORT:-7865}" \
    --checkpoint_path "$CHECKPOINT_DIR" \
    --device_id "$DEVICE_ID" \
    --bf16 "$BF16" \
    --torch_compile "$TORCH_COMPILE" \
    --cpu_offload "$CPU_OFFLOAD" \
    --overlapped_decode "$OVERLAPPED_DECODE"
fi

export CHECKPOINT_DIR
export DEVICE_ID
export BF16
export TORCH_COMPILE
export CPU_OFFLOAD
export OVERLAPPED_DECODE

exec python3 infer-api.py
