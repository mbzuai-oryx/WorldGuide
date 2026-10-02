"""Training-forward-aligned mode-2 memory evaluation with Qwen-planned steps.

Same strict preset as run_eval_mem_multi_gpu.py (model_type=bi, mode2 memory,
generated_prev references), but step captions are produced online by the Qwen
planner from the generated history instead of being read from the manifest.
See run_step_prompts_v5_qwen_planner_multi_gpu.py for the planner options.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyvideo import run_eval_mem_multi_gpu as mem_preset
from hyvideo import run_step_prompts_v5_qwen_planner_multi_gpu as planner_multi_gpu


def _with_planner_memory_preset(argv: list[str]) -> list[str]:
    if any(token in ("-h", "--help") for token in argv):
        return argv
    if any(mem_preset._option_name(token) == "--prompt_manifest_path" for token in argv):
        raise ValueError(
            "run_eval_mem_qwen_planner_multi_gpu.py generates step prompts with the planner; "
            "remove --prompt_manifest_path."
        )
    # Keep outputs apart from manifest-caption runs when no prefix was supplied.
    if not any(mem_preset._option_name(token) == "--output_prefix" for token in argv):
        argv = [*argv, "--output_prefix", "mem_planner_"]
    return mem_preset._with_memory_preset(argv)


def main() -> None:
    planner_multi_gpu.main(_with_planner_memory_preset(sys.argv[1:]))


if __name__ == "__main__":
    main()


'''
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate worldguide

  cd worldGuide

  export MIOPEN_USER_DB_PATH=.cache/miopen
  export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

  python3 hyvideo/run_eval_mem_qwen_planner_multi_gpu.py \
    --manifest_path data/test_manifest.json \
    --planner_model_path ckpts/worldguide_planner \
    --output_root ./outputs/testdata_eval_mem_qwen_planner_ckpt2385_33frames \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --max_parallel_jobs 8 \
    --launch_stagger_sec 0 \
    --master_port_base 29500 \
    --resume true \
    --model_path ckpts/HunyuanVideo-1.5 \
    --num_inference_steps 10 \
    --guidance_scale 7.5 \
    --flow_shift 5.0 \
    --memory_frames 20 \
    --temporal_context_size 12 \
    --memory_frame_policy recent \
    --planner_history_k 3 \
    --max_planned_steps 40 \
    --resolution 480p \
    --aspect_ratio 16:9 \
    --height 480 \
    --width 832 \
    --output_fps 16 \
    --dtype bf16 \
    --seed 3208

  Fixed internally (same as run_eval_mem_multi_gpu.py, do not pass):
  step_video_length=33 chunk_latent_frames=9 model_type=bi origami_memory_mode=mode2
  origami_memory_steps=2 origami_memory_policy=latest_k origami_memory_blend=1.0
  continuity_mode=stateless_prompt_ref reference_source=generated_prev
  reference_frame_mode=last prompt_memory_steps=0
'''
