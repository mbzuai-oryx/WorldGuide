#!/usr/bin/env bash
# WorldGuide: Precompute Normalized Manifest and Cached Features
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

RAW_MANIFEST="${RAW_MANIFEST:-data/train_manifest.json}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-data/train_manifest_precomputed.json}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-data/origami_feature_cache}"
MODEL_PATH="${MODEL_PATH:-ckpts/HunyuanVideo-1.5}"
DEVICE="${DEVICE:-cuda}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
TARGET_FPS="${TARGET_FPS:-16}"
HISTORY_KEEP="${HISTORY_KEEP:-3}"

python3 trainer/dataset/origami_step_precompute.py \
  --raw-manifest "${RAW_MANIFEST}" \
  --output-manifest "${OUTPUT_MANIFEST}" \
  --feature-cache-dir "${FEATURE_CACHE_DIR}" \
  --model-path "${MODEL_PATH}" \
  --device "${DEVICE}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --target-fps "${TARGET_FPS}" \
  --history-keep "${HISTORY_KEEP}" \
  --reference-mode latest_past_or_global \
  --shard-strategy balanced_duration \
  --dist-timeout-seconds 7200 \
  --save-every 100 \
  "$@"
