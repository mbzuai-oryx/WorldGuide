"""Training-forward-aligned mode-2 memory evaluation with online Qwen planning.

Qwen generates each step from the goal and generated history by default.
Use --step_source manifest to replay reference captions instead.
See hyvideo/QWEN_PLANNER_EVAL.md for test-manifest commands.

This is a strict preset over run_step_prompts_v5_multi_gpu.py.  In particular,
it forces model_type=bi so the tiered memory-compression tokens are consumed by
the same full-window transformer path used during training.  The older V5
launcher defaulted to model_type=ar, whose forward_vision path computed those
tokens but did not feed them to attention.

The preset evaluates free-running generated history.  Historical vision states
come from the RGB reference used for each generated step, matching the training
token ordering while intentionally replacing gold history with generated history.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyvideo import run_step_prompts_v5_multi_gpu as v5_multi_gpu


DEFAULT_ACTION_CKPT = (
    "ckpts/WorldGuide-Ckpt/transformer/diffusion_pytorch_model.safetensors"
)

FORCED_OPTIONS = {
    "--step_video_length": "33",
    "--chunk_latent_frames": "9",
    "--model_type": "bi",
    "--origami_memory_mode": "mode2",
    "--origami_memory_steps": "2",
    "--origami_memory_policy": "latest_k",
    "--origami_memory_blend": "1.0",
    "--continuity_mode": "stateless_prompt_ref",
    "--reference_source": "generated_prev",
    "--reference_frame_mode": "last",
    "--prompt_memory_steps": "0",
}


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


def _reject_forced_option_overrides(argv: list[str]) -> None:
    supplied = {_option_name(token) for token in argv if token.startswith("--")}
    conflicts = sorted(supplied.intersection(FORCED_OPTIONS))
    if conflicts:
        values = ", ".join(
            f"{option}={FORCED_OPTIONS[option]}" for option in conflicts
        )
        raise ValueError(
            "run_eval_mem_multi_gpu.py fixes training-aligned memory options; "
            f"remove these overrides: {values}"
        )


def _with_memory_preset(argv: list[str]) -> list[str]:
    if any(token in ("-h", "--help") for token in argv):
        return argv

    _reject_forced_option_overrides(argv)
    preset_args = [
        item
        for option, value in FORCED_OPTIONS.items()
        for item in (option, value)
    ]
    if not any(_option_name(token) == "--output_prefix" for token in argv):
        source_parser = argparse.ArgumentParser(add_help=False)
        source_parser.add_argument("--step_source", default="qwen_planner")
        step_source = source_parser.parse_known_args(argv)[0].step_source
        prefix = "mem_planner_" if step_source == "qwen_planner" else "mem_"
        preset_args.extend(["--output_prefix", prefix])
    if not any(_option_name(token) == "--action_ckpt" for token in argv):
        preset_args.extend(["--action_ckpt", DEFAULT_ACTION_CKPT])
    return [*argv, *preset_args]


def main() -> None:
    v5_multi_gpu.main(_with_memory_preset(sys.argv[1:]))


if __name__ == "__main__":
    main()


'''
srun --overlap --jobid=YOUR_ACTIVE_GPU_JOB_ID --pty bash

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate worldguide

cd worldGuide

export MIOPEN_USER_DB_PATH=.cache/miopen
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p "$MIOPEN_USER_DB_PATH"

python3 hyvideo/run_eval_mem_multi_gpu.py \
    --step_source manifest \
    --manifest_path data/Video_CraftBench_v0.1/video_craftbench_caption.json \
    --prompt_manifest_path data/Video_CraftBench_v0.1/video_craftbench_caption.json \
    --output_root outputs/video_craftbench_eval_mem_ckpt2385_33frames_all \
    --output_prefix mem_ \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --max_parallel_jobs 8 \
    --launch_stagger_sec 0 \
    --master_port_base 29500 \
    --resume true \
    --model_path ckpts/HunyuanVideo-1.5 \
    --action_ckpt ckpts/WorldGuide-Ckpt/transformer/diffusion_pytorch_model.safetensors \
    --num_inference_steps 10 \
    --guidance_scale 7.5 \
    --flow_shift 5.0 \
    --memory_frames 20 \
    --temporal_context_size 12 \
    --memory_frame_policy recent \
    --resolution 480p \
    --aspect_ratio 16:9 \
    --height 480 \
    --width 832 \
    --output_fps 16 \
    --dtype bf16 \
    --seed 3208

  Do not add the following options manually because run_eval_mem_multi_gpu.py fixes them internally:

  step_video_length=33
  chunk_latent_frames=9
  model_type=bi
  origami_memory_mode=mode2
  origami_memory_steps=2
  reference_source=generated_prev
  continuity_mode=stateless_prompt_ref
  reference_frame_mode=last


'''
