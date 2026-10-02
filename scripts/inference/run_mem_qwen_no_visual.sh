#!/usr/bin/env bash
# Run inside an existing allocation. The observed 256 GiB job OOM-killed eight
# simultaneous workers; start two at a time, with a gap between model loads.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export MIOPEN_USER_DB_PATH=.cache/miopen
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MASTER_PORT=29500
export PYTHONFAULTHANDLER=1
mkdir -p "$MIOPEN_USER_DB_PATH"

exec python3 \
  hyvideo/run_eval_mem_qwen_planner_no_visual_multi_gpu.py \
  --manifest_path data/test_manifest.json \
  --output_root ./outputs/testdata_mem_qwen_no_visual_33frames \
  --gpu_ids 0,1 \
  --max_parallel_jobs 2 \
  --launch_stagger_sec 60 \
  --master_port_base 29500 \
  --resume true \
  --model_path ckpts/HunyuanVideo-1.5 \
  --action_ckpt ckpts/WorldGuide-Ckpt/transformer/diffusion_pytorch_model.safetensors \
  --planner_model_path ckpts/WorldGuide-Ckpt/text_encoder/llm \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --flow_shift 5.0 \
  --memory_frames 20 \
  --temporal_context_size 12 \
  --memory_frame_policy recent \
  --planner_history_k 3 \
  --planner_offload true \
  --max_planned_steps 70 \
  --stop_on_task_completed true \
  --save_summary_json true \
  --resolution 480p \
  --aspect_ratio 16:9 \
  --height 480 \
  --width 832 \
  --output_fps 16 \
  --dtype bf16 \
  --seed 3208 \
  "$@"
