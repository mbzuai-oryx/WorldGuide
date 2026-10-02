#!/usr/bin/env bash
# V5 Qwen ablation on 64 GiB GPUs: keep Hunyuan resident between clips.
# Qwen is still offloaded before Hunyuan runs. Uses the existing pipeline flags.
# This is a mitigation for native transfer/cleanup crashes, not a verified fix.
# Extra arguments override defaults. Run inside an existing GPU allocation.
set -euo pipefail

V5_ABLATION_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$V5_ABLATION_SCRIPT_DIR/run_v5_qwen_45frames_shard1.sh" \
  --output_root ./outputs/testdata_v5_qwen_45frames \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --max_parallel_jobs 8 \
  --offloading false \
  --group_offloading false \
  --planner_offload true \
  --planner_video_backend ffmpeg \
  --start_index 103 \
  --end_index 206 \
  "$@"
