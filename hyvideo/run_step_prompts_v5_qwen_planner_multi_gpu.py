"""Explicit Qwen-planner entry point for the shared V5 multi-GPU launcher."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyvideo import run_step_prompts_v5_multi_gpu as v5_multi_gpu

PLANNER_GENERATOR = v5_multi_gpu.PLANNER_GENERATOR


def build_parser():
    parser = v5_multi_gpu.build_parser()
    parser.set_defaults(step_source="qwen_planner", reference_source="generated_prev", output_prefix="planner_",
                        planner_video_backend="ffmpeg")
    return parser


def build_generation_command(args, video_id: str, output_dir: Path, master_port: int) -> list[str]:
    if args.step_source != "qwen_planner":
        raise ValueError("This entry point requires --step_source qwen_planner.")
    return v5_multi_gpu.build_generation_command(args, video_id, output_dir, master_port)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    if args.step_source != "qwen_planner":
        raise ValueError("Use run_step_prompts_v5_multi_gpu.py for --step_source manifest.")
    # The shared launcher's parser has its own defaults. Forward this ablation's
    # choice explicitly; --planner_video_backend qwen remains an opt-in.
    v5_multi_gpu.main([*argv, "--planner_video_backend", args.planner_video_backend])


if __name__ == "__main__":
    main()


"""
cd worldGuide
export PYTHONPATH=$(pwd):$PYTHONPATH

PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 hyvideo/run_step_prompts_v5_qwen_planner_multi_gpu.py \
  --manifest_path data/test_manifest.json \
  --planner_model_path ckpts/WorldGuide-Ckpt/text_encoder/llm \
  --output_root ./outputs/worldguide_v5_qwen_planner_testdata \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --max_parallel_jobs 8 \
  --launch_stagger_sec 0 \
  --resume true \
  --model_path ckpts/HunyuanVideo-1.5 \
  --action_ckpt ckpts/WorldGuide-Ckpt/transformer/diffusion_pytorch_model.safetensors \
  --step_video_length 33 \
  --num_inference_steps 30 \
  --guidance_scale 6.0 \
  --flow_shift 5.0 \
  --reference_source generated_prev \
  --continuity_mode stateless_prompt_ref \
  --reference_frame_mode last \
  --chunk_latent_frames 9 \
  --memory_frames 20 \
  --temporal_context_size 12 \
  --memory_frame_policy recent \
  --model_type bi \
  --origami_memory_mode mode2 \
  --origami_memory_steps 2 \
  --origami_memory_policy latest_k \
  --origami_memory_blend 1.0 \
  --planner_history_k 3 \
  --max_planned_steps 40 \
  --resolution 480p \
  --aspect_ratio 16:9 \
  --height 480 \
  --width 832 \
  --output_fps 16 \
  --dtype bf16 \
  --seed 3208
"""
