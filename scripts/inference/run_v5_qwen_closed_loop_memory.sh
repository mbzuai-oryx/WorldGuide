#!/usr/bin/env bash
# Trained Qwen -> execute one action with mode-2 memory -> observe -> replan.
# Run inside an existing eight-GPU allocation with enough host RAM (512 GiB
# allocations were used for these eight-worker evaluations).
# Extra arguments override defaults, e.g. --start_index 0 --end_index 103.
set -euo pipefail

V5_CLOSED_LOOP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the ablation's existing native-crash mitigations. Serialization may
# reduce throughput; it does not select devices or change planner inputs.
export PYTORCH_ALLOC_CONF="${V5_RETRY_ALLOC_CONF:-expandable_segments:False}"
export AMD_SERIALIZE_KERNEL="${V5_RETRY_SERIALIZE_KERNEL:-3}"
export AMD_SERIALIZE_COPY="${V5_RETRY_SERIALIZE_COPY:-3}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# The resident script supplies the manifest, trained checkpoints, 45-frame
# clips, 30 denoising iterations, seed 3208, and 70-step cap. Override its
# mutable shard defaults and explicitly select the visual closed loop.
exec bash "$V5_CLOSED_LOOP_SCRIPT_DIR/run_v5_qwen_45frames_resident.sh" \
  --step_source qwen_planner \
  --planner_visual_feedback true \
  --planner_history_k 3 \
  --planner_nframes 8 \
  --planner_video_backend ffmpeg \
  --planner_offload true \
  --planner_retry_empty true \
  --reference_source generated_prev \
  --reference_frame_mode last \
  --continuity_mode stateless_prompt_ref \
  --prompt_memory_steps 0 \
  --model_type bi \
  --origami_memory_mode mode2 \
  --origami_memory_steps 2 \
  --origami_memory_policy latest_k \
  --origami_memory_blend 1.0 \
  --offloading false \
  --group_offloading false \
  --vae_decode_offload_transformer true \
  --ids_file "" \
  --start_index 0 \
  --end_index 206 \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --max_parallel_jobs 8 \
  --output_root ./outputs/testdata_v5_qwen_45frames \
  --save_summary_json true \
  --resume true \
  "$@"
