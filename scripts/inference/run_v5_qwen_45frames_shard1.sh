#!/usr/bin/env bash
# Run inside the existing allocation; no allocation or inference smoke test here.
# Extra arguments override defaults, e.g. --start_index 2 --end_index 3.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export MIOPEN_USER_DB_PATH=.cache/miopen
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
# Allow an explicit allocator setting for ablation runtime diagnostics.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MASTER_PORT=29500
export PYTHONFAULTHANDLER=1
export HYVIDEO_DEBUG_STAGES=1
mkdir -p "$MIOPEN_USER_DB_PATH"

exec python3 \
  hyvideo/run_step_prompts_v5_qwen_planner_multi_gpu.py \
  --manifest_path data/test_manifest.json \
  --output_root ./outputs/testdata_v5_qwen_45frames \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --max_parallel_jobs 8 \
  --launch_stagger_sec 20 \
  --master_port_base 29500 \
  --resume true \
  --model_path ckpts/HunyuanVideo-1.5 \
  --action_ckpt ckpts/worldguide_action/diffusion_pytorch_model.safetensors \
  --planner_model_path ckpts/worldguide_planner \
  --model_type bi \
  --step_video_length 45 \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --flow_shift 5.0 \
  --reference_source generated_prev \
  --continuity_mode stateless_prompt_ref \
  --reference_frame_mode last \
  --prompt_memory_steps 0 \
  --chunk_latent_frames 12 \
  --memory_frames 20 \
  --temporal_context_size 12 \
  --memory_frame_policy recent \
  --origami_memory_mode mode2 \
  --origami_memory_steps 2 \
  --origami_memory_policy latest_k \
  --origami_memory_blend 1.0 \
  --planner_history_k 3 \
  --planner_nframes 8 \
  --planner_video_backend ffmpeg \
  --planner_offload true \
  --max_planned_steps 70 \
  --stop_on_task_completed true \
  --resolution 480p \
  --aspect_ratio 16:9 \
  --height 480 \
  --width 832 \
  --output_fps 16 \
  --dtype bf16 \
  --seed 3208 \
  --start_index 103 \
  --end_index 206 \
  "$@"
