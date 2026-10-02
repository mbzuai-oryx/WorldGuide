"""Planner prompts and CLI shared by the launcher and generation worker.

Keep this module free of model imports so launchers and CPU checks do not need
to load the video pipeline. Visual-feedback prompts match YUME_VLA's
vla_dataset.py; the action-history ablation omits observation references.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any


TASK_COMPLETED_TOKEN = "<|Task Completed|>"
PLANNER_SYSTEM_PROMPT = (
    "You are a step-by-step action planner. "
    "Given observation images showing the current state and a task goal, "
    "predict the NEXT single action step. "
    "Output ONLY the action description as one clear sentence."
)
ACTION_HISTORY_PLANNER_SYSTEM_PROMPT = (
    "You are a step-by-step action planner. "
    "Given a task goal and recent executed action descriptions, "
    "predict the NEXT single action step. "
    "Output ONLY the action description as one clear sentence."
)
DEFAULT_PLANNER_MODEL_PATH = str(
    Path(__file__).resolve().parents[2] / "YUME_VLA/Checkpoint/Phase1_Planner4/final"
)
DEFAULT_MAX_PLANNED_STEPS = 40
CATEGORY_RE = re.compile(r"Category:\s*([^.]+)\.", re.IGNORECASE)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def resolve_category(row: dict[str, Any]) -> str:
    category = str(row.get("category") or "").strip()
    if category:
        return category
    match = CATEGORY_RE.search(str(row.get("task") or row.get("query") or ""))
    return match.group(1).strip() if match else ""


def build_task_goal(row: dict[str, Any]) -> str:
    """Use only goal fields; never infer a goal from the reference captions."""
    task = str(row.get("task") or "").strip() or str(row.get("query") or "").strip()
    work_subject = str(row.get("work_subject") or "").strip()
    if len(work_subject) < 5 and task:
        return f"A step-by-step tutorial based on {task}."
    if work_subject:
        category = resolve_category(row)
        tutorial = f"{category} tutorial" if category else "tutorial"
        return f"A step-by-step {tutorial} showing how to make {work_subject.lower()}."
    raise ValueError(f"video_id={row.get('video_id')}: provide task, query, or work_subject for the planner goal.")


def build_planner_prompt(
    task_goal: str,
    history: list[str],
    step_number: int,
    is_env_preview: bool,
    max_history_text: int,
    visual_feedback: bool = True,
) -> str:
    parts = [f"Task goal: {task_goal}"]
    if not visual_feedback and step_number == 1:
        parts.append("\nThis is the very first step. Based on the task goal, predict the first action (Step 1):")
    elif visual_feedback and is_env_preview:
        parts.append("The video clip above shows a 1-second preview of the starting environment.")
        parts.append("\nThis is the very first step. Based on the environment preview, predict the first action (Step 1):")
    else:
        if history:
            parts.append("\nRecent steps executed:")
            # Training uses 0 to retain all history.
            history_to_keep = history[-max_history_text:] if max_history_text > 0 else history
            start_step = step_number - len(history_to_keep)
            for i, caption in enumerate(history_to_keep):
                parts.append(f"  Step {start_step + i}: {caption}")
        context = "recent visual clips and action history" if visual_feedback else "task goal and action history"
        parts.append(
            f"\nBased on the {context} above, "
            f"predict the exact action caption for the VERY NEXT step (Step {step_number}):"
        )
    return "\n".join(parts)


def clean_planner_prediction(text: str, processor, *, collapse_whitespace: bool = True) -> str:
    """Remove transport tokens while preserving the trained completion token."""
    if not text:
        return ""
    tokenizer = getattr(processor, "tokenizer", None)
    specials = set(getattr(tokenizer, "all_special_tokens", []) or [])
    specials.discard(TASK_COMPLETED_TOKEN)
    for token in sorted(specials, key=len, reverse=True):
        if token:
            text = text.replace(token, " ")
    text = re.sub(
        r"<\|[^|]+?\|>",
        lambda match: match.group(0) if match.group(0) == TASK_COMPLETED_TOKEN else " ",
        text,
    )
    return " ".join(text.strip().split()) if collapse_whitespace else text.strip()


def split_completion(prediction: str) -> tuple[str, bool]:
    completed = TASK_COMPLETED_TOKEN in prediction
    action = " ".join(prediction.replace(TASK_COMPLETED_TOKEN, " ").split())
    return action, completed


def add_planner_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--planner_model_path", type=str, default=DEFAULT_PLANNER_MODEL_PATH,
                        help="Trained Qwen planner checkpoint (HF directory); defaults to Phase1_Planner4/final.")
    parser.add_argument("--planner_processor_path", type=str, default=None,
                        help="Optional processor directory; defaults to --planner_model_path.")
    parser.add_argument("--planner_visual_feedback", type=str_to_bool, nargs="?", const=True, default=True,
                        help="If false, Qwen uses only the goal and predicted action history, including at step 1.")
    parser.add_argument("--planner_history_k", type=int, default=3,
                        help="Previous predicted captions in the prompt (training: 3; 0 keeps all history).")
    parser.add_argument("--planner_nframes", type=int, default=8)
    parser.add_argument("--planner_video_backend", choices=["qwen", "ffmpeg"], default="qwen",
                        help="qwen uses its default reader; ffmpeg isolates native video decoding in subprocesses.")
    parser.add_argument("--planner_max_pixels", type=int, default=65536)
    parser.add_argument("--planner_min_pixels", type=int, default=16384)
    parser.add_argument("--planner_max_new_tokens", type=int, default=128)
    parser.add_argument("--planner_retry_empty", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Closed-loop ablation: retry an empty prediction once, blocking transport tokens at the first new token while allowing Task Completed.")
    parser.add_argument("--planner_dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--planner_offload", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Keep the planner on CPU while HunyuanVideo runs.")
    parser.add_argument("--vae_decode_offload_transformer", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Qwen ablation only: park the resident transformer on CPU during VAE decode, then restore it.")
    parser.add_argument("--max_planned_steps", type=int, default=DEFAULT_MAX_PLANNED_STEPS,
                        help="Step cap independent of manifest total_steps; --video_length overrides it.")
    parser.add_argument("--stop_on_task_completed", type=str_to_bool, nargs="?", const=True, default=True)
    return parser


def validate_planner_args(args) -> None:
    if args.prompt_manifest_path:
        raise ValueError("Qwen generates every step prompt; remove --prompt_manifest_path (or use --step_source manifest in the launcher).")
    if args.continuity_mode == "last":
        args.continuity_mode = "stateless_prompt_ref"
        args.reference_frame_mode = "last"
    if args.continuity_mode != "stateless_prompt_ref" or args.reference_source != "generated_prev":
        raise ValueError("Planner mode requires --continuity_mode stateless_prompt_ref and --reference_source generated_prev.")
    if args.planner_retry_empty and getattr(args, "step_source", "qwen_planner") != "qwen_planner":
        raise ValueError("planner_retry_empty is supported only for the closed-loop Qwen ablation.")
    if args.vae_decode_offload_transformer:
        if args.offloading or args.group_offloading or args.model_type != "bi":
            raise ValueError("vae_decode_offload_transformer requires offloading=false, group_offloading=false, and model_type=bi.")
        if getattr(args, "step_source", "qwen_planner") != "qwen_planner":
            raise ValueError("vae_decode_offload_transformer is supported only for the closed-loop Qwen ablation.")
    for name in ("step_video_length", "planner_nframes", "planner_max_pixels", "planner_min_pixels",
                 "planner_max_new_tokens", "max_planned_steps", "memory_frames", "temporal_context_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.video_length is not None and args.video_length <= 0:
        raise ValueError("video_length must be positive when provided.")
    if args.planner_max_pixels < args.planner_min_pixels:
        raise ValueError("planner_max_pixels must be >= planner_min_pixels.")
    if args.planner_history_k < 0 or args.prompt_memory_steps < 0:
        raise ValueError("planner_history_k and prompt_memory_steps must be >= 0.")
    if args.origami_memory_mode == "mode2":
        if args.origami_memory_policy != "latest_k" or args.origami_memory_steps <= 0:
            raise ValueError("mode2 requires origami_memory_policy=latest_k and positive origami_memory_steps.")
    if not args.planner_model_path or not os.path.isdir(args.planner_model_path):
        raise FileNotFoundError(f"Missing trained planner checkpoint: {args.planner_model_path}; set --planner_model_path.")
    if args.planner_processor_path and not os.path.isdir(args.planner_processor_path):
        raise FileNotFoundError(f"Missing planner_processor_path: {args.planner_processor_path}")


def planner_command_args(args) -> list[str]:
    options = (
        "planner_model_path", "planner_visual_feedback", "planner_history_k", "planner_nframes", "planner_video_backend", "planner_max_pixels",
        "planner_min_pixels", "planner_max_new_tokens", "planner_dtype", "planner_offload", "vae_decode_offload_transformer",
        "max_planned_steps", "stop_on_task_completed", "planner_retry_empty",
    )
    command = []
    for name in options:
        value = getattr(args, name)
        command.extend([f"--{name}", str(value).lower() if isinstance(value, bool) else str(value)])
    if args.planner_processor_path:
        command.extend(["--planner_processor_path", args.planner_processor_path])
    return command
