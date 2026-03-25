#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-./configs/flash_attention_dit.json}"
CHECKPOINT_PATH="${2:?Please provide a checkpoint path}"
NORM_PATH="${3:-}"

ARGS=(
  --config_file "${CONFIG_FILE}"
  --resume "${CHECKPOINT_PATH}"
  --eval_only
  --output_dir ./outputs
)

if [[ -n "${NORM_PATH}" ]]; then
  ARGS+=(--normalization_file "${NORM_PATH}")
fi

python -m flash_battery.train "${ARGS[@]}"
