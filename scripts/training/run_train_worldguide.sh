#!/usr/bin/env bash
# WorldGuide: Distributed Training with Visual Memory
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Checkpoints and Data
export MODEL_PATH="${MODEL_PATH:-ckpts/HunyuanVideo-1.5}"
export NORMALIZED_MANIFEST="${NORMALIZED_MANIFEST:-data/train_manifest_precomputed.json}"
export RAW_MANIFEST="${RAW_MANIFEST:-data/train_manifest.json}"
export ORIGAMI_FEATURE_SOURCE="${ORIGAMI_FEATURE_SOURCE:-precomputed}"
export LOAD_FROM_DIR="${LOAD_FROM_DIR:-ckpts/HunyuanVideo-1.5/transformer/480p_i2v}"
export AR_ACTION_LOAD_FROM_DIR="${AR_ACTION_LOAD_FROM_DIR:-ckpts/worldguide_action/diffusion_pytorch_model.safetensors}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/worldguide_train}"

# Training Schedule
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500}"
export LOG_STEPS="${LOG_STEPS:-10}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

# Distributed Topology
export NNODES="${NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29612}"
export SP_SIZE="${SP_SIZE:-1}"

bash scripts/training/hyvideo15/run_ar_hunyuan_origami_steps.sh "$@"
