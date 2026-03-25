#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-./configs/flash_attention_dit.json}"

python -m flash_battery.train \
  --config_file "${CONFIG_FILE}" \
  --output_dir ./outputs
