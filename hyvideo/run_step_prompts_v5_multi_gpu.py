"""Multi-GPU V5 evaluation with online Qwen planning by default.

Use --step_source manifest to replay the previous manifest-caption evaluation.
See hyvideo/QWEN_PLANNER_EVAL.md for test-manifest commands.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyvideo.qwen_planner_utils import add_planner_arguments, planner_command_args, validate_planner_args
from hyvideo.open_loop_planner_utils import frame_schedule
from hyvideo.inference_checks import check_inference_checkpoints

V5_GENERATOR = REPO_ROOT / "hyvideo" / "generate_continuous_step_promptsV5.py"
PLANNER_GENERATOR = REPO_ROOT / "hyvideo" / "generate_continuous_step_promptsV5_qwen_planner.py"
OPEN_LOOP_GENERATOR = REPO_ROOT / "hyvideo" / "generate_open_loop_qwen_planner.py"
NO_AMDSMI_PATH = REPO_ROOT / "hyvideo" / "runtime_no_amdsmi"


def str_to_bool(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("true", "1", "yes", "on"):
            return True
        if value in ("false", "0", "no", "off"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def _default_model_path() -> str:
    return os.environ.get("MODEL_PATH", "ckpts/HunyuanVideo-1.5")


def _default_action_ckpt() -> str:
    return os.environ.get("ACTION_CKPT", "ckpts/worldguide_action/diffusion_pytorch_model.safetensors")


def sanitize_filename(value: str) -> str:
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def bool_string(value: bool) -> str:
    return "true" if value else "false"


def parse_id_list(ids_text_path: str) -> list[str]:
    text = Path(ids_text_path).read_text(encoding="utf-8")
    normalized = text.replace("\n", ",")
    raw_tokens = [token.strip() for token in normalized.split(",")]
    deduped: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    if not deduped:
        raise ValueError(f"No valid video IDs found in file: {ids_text_path}")
    return deduped


def load_manifest_video_ids(manifest_path: str) -> list[str]:
    with open(manifest_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if isinstance(payload, dict):
        for key in ("records", "data", "items", "videos", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    video_ids: list[str] = []
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            continue
        if video_id in seen:
            continue
        seen.add(video_id)
        video_ids.append(video_id)
    if not video_ids:
        raise ValueError(f"No video_id values found in manifest: {manifest_path}")
    return video_ids


def select_video_ids(
    manifest_path: str,
    ids_file: str | None,
    start_index: int = 0,
    end_index: int | None = None,
) -> list[str]:
    manifest_video_ids = load_manifest_video_ids(manifest_path)
    if not ids_file:
        selected_video_ids = manifest_video_ids
    else:
        requested_video_ids = parse_id_list(ids_file)
        manifest_ids_set = set(manifest_video_ids)
        missing_video_ids = [
            video_id for video_id in requested_video_ids if video_id not in manifest_ids_set
        ]
        if missing_video_ids:
            preview = ", ".join(missing_video_ids[:10])
            suffix = "" if len(missing_video_ids) <= 10 else f", ... (+{len(missing_video_ids) - 10} more)"
            raise ValueError(
                f"{len(missing_video_ids)} IDs from {ids_file} were not found in "
                f"{manifest_path}: {preview}{suffix}"
            )
        selected_video_ids = requested_video_ids

    if start_index < 0:
        raise ValueError("start_index must be >= 0.")
    if end_index is not None and end_index < start_index:
        raise ValueError("end_index must be >= start_index.")
    return selected_video_ids[start_index:end_index]


def parse_gpu_ids(gpu_ids_arg: str) -> list[str]:
    gpu_ids = [token.strip() for token in gpu_ids_arg.split(",") if token.strip()]
    if not gpu_ids:
        raise ValueError("No GPU IDs provided.")
    deduped: list[str] = []
    seen: set[str] = set()
    for gpu_id in gpu_ids:
        if gpu_id in seen:
            continue
        seen.add(gpu_id)
        deduped.append(gpu_id)
    return deduped


def build_worker_env(gpu_id: str) -> dict[str, str]:
    env = os.environ.copy()
    repo_root = str(REPO_ROOT)
    no_amdsmi_path = str(NO_AMDSMI_PATH)
    current_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [no_amdsmi_path, repo_root]
    if current_pythonpath:
        pythonpath_entries.append(current_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    # Use HIP alone for worker selection, even when the parent shell or Slurm
    # exported competing GPU masks. Keep the parent environment unchanged.
    for name in tuple(env):
        if (
            name.endswith("_VISIBLE_DEVICES") and name != "HIP_VISIBLE_DEVICES"
        ) or name in ("GPU_DEVICE_ORDINAL", "CUDA_DEVICE_ORDER"):
            env.pop(name)
    env["HIP_VISIBLE_DEVICES"] = gpu_id
    env["MASTER_ADDR"] = "127.0.0.1"
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("OMP_NUM_THREADS", "1")
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    return env


def is_completed_video(
    output_dir: Path, final_output_name: str, step_source: str | None = None,
    planner_visual_feedback: bool | None = None,
    expected_frame_count: int | None = None,
    expected_output_fps: int | None = None,
) -> bool:
    summary_path = output_dir / "step_prompt_summary.json"
    final_path = output_dir / final_output_name
    if not summary_path.exists() or not final_path.exists():
        return False
    if final_path.stat().st_size <= 0:
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as fp:
            summary = json.load(fp)
    except Exception:
        return False
    if step_source is not None and summary.get("step_source", "manifest") != step_source:
        return False
    if step_source == "qwen_planner" and summary.get("stop_reason") not in ("task_completed", "plan_limit_reached"):
        return False
    if step_source == "qwen_planner" and planner_visual_feedback is not None:
        # Summaries written before this ablation always used visual feedback.
        if summary.get("planner_visual_feedback", True) != planner_visual_feedback:
            return False
    if step_source == "qwen_open_loop":
        if summary.get("stop_reason") != "plan_executed":
            return False
        if expected_frame_count is not None and summary.get("final_written_frames") != expected_frame_count:
            return False
        if expected_output_fps is not None and summary.get("output_fps") != expected_output_fps:
            return False
    return bool(summary.get("final_output_path") or summary.get("steps"))


def should_skip_video(
    output_dir: Path, final_output_name: str, resume: bool, step_source: str | None = None,
    planner_visual_feedback: bool | None = None,
    expected_frame_count: int | None = None,
    expected_output_fps: int | None = None,
) -> bool:
    return resume and is_completed_video(output_dir, final_output_name, step_source, planner_visual_feedback, expected_frame_count, expected_output_fps)


def build_generation_command(args, video_id: str, output_dir: Path, master_port: int) -> list[str]:
    use_planner = args.step_source in ("qwen_planner", "qwen_open_loop")
    generator = OPEN_LOOP_GENERATOR if args.step_source == "qwen_open_loop" else PLANNER_GENERATOR if use_planner else V5_GENERATOR
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--nnodes=1",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(master_port),
        str(generator),
        "--manifest_path",
        args.manifest_path,
        f"--video_id={video_id}",
        "--output_path",
        str(output_dir),
        "--action_ckpt",
        args.action_ckpt,
        "--model_path",
        args.model_path,
        "--step_video_length",
        str(args.step_video_length),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--chunk_latent_frames",
        str(args.chunk_latent_frames),
        "--reference_frame_mode",
        args.reference_frame_mode,
        "--reference_source",
        args.reference_source,
        "--continuity_mode",
        args.continuity_mode,
        "--prompt_memory_steps",
        str(args.prompt_memory_steps),
        "--seed_schedule",
        args.seed_schedule,
        "--memory_frames",
        str(args.memory_frames),
        "--temporal_context_size",
        str(args.temporal_context_size),
        "--memory_frame_policy",
        args.memory_frame_policy,
        "--origami_memory_mode",
        args.origami_memory_mode,
        "--origami_memory_steps",
        str(args.origami_memory_steps),
        "--origami_memory_policy",
        args.origami_memory_policy,
        "--origami_memory_blend",
        str(args.origami_memory_blend),
        "--model_type",
        args.model_type,
        "--resolution",
        args.resolution,
        "--aspect_ratio",
        args.aspect_ratio,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--output_fps",
        str(args.output_fps),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--offloading",
        bool_string(args.offloading),
        "--group_offloading",
        bool_string(args.group_offloading),
        "--final_output_name",
        args.final_output_name,
        "--save_summary_json",
        bool_string(args.save_summary_json),
        "--stitch_drop_first_after_step1",
        bool_string(args.stitch_drop_first_after_step1),
    ]
    if args.image_path:
        cmd.extend(["--image_path", args.image_path])
    if args.prompt_manifest_path and not use_planner:
        cmd.extend(["--prompt_manifest_path", args.prompt_manifest_path])
    if args.initial_preview_clip:
        cmd.extend(["--initial_preview_clip", args.initial_preview_clip])
    if args.guidance_scale is not None:
        cmd.extend(["--guidance_scale", str(args.guidance_scale)])
    if args.flow_shift is not None:
        cmd.extend(["--flow_shift", str(args.flow_shift)])
    if args.video_length is not None:
        cmd.extend(["--video_length", str(args.video_length)])
    if args.negative_prompt:
        cmd.extend(["--negative_prompt", args.negative_prompt])
    if args.enable_torch_compile:
        cmd.extend(["--enable_torch_compile", "true"])
    if args.use_sageattn:
        cmd.extend(["--use_sageattn", "true", "--sage_blocks_range", args.sage_blocks_range])
    if args.use_vae_parallel:
        cmd.extend(["--use_vae_parallel", "true"])
    if args.use_fp8_gemm:
        cmd.extend(["--use_fp8_gemm", "true", "--quant_type", args.quant_type, "--include_patterns", args.include_patterns])
    if args.few_step:
        cmd.extend(["--few_step", "true"])
    if args.save_step_latents:
        cmd.extend(["--save_step_latents", "true"])
    if args.transformer_resident_ar_rollout:
        cmd.extend(["--transformer_resident_ar_rollout", "true"])
    if use_planner:
        cmd.extend(planner_command_args(args))
    if args.step_source == "qwen_open_loop":
        cmd.extend(["--target_duration_seconds", str(args.target_duration_seconds)])
    return cmd


def launch_job(args, video_id: str, gpu_id: str, slot_index: int, logs_dir: Path) -> dict[str, Any]:
    output_dir = Path(args.output_root) / f"{args.output_prefix}{sanitize_filename(video_id)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    master_port = args.master_port_base + slot_index
    cmd = build_generation_command(args, video_id=video_id, output_dir=output_dir, master_port=master_port)
    env = build_worker_env(gpu_id)
    log_path = logs_dir / f"{sanitize_filename(video_id)}.log"
    log_handle = open(log_path, "a", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "video_id": video_id,
        "gpu_id": gpu_id,
        "slot_index": slot_index,
        "master_port": master_port,
        "process": process,
        "log_handle": log_handle,
        "log_path": str(log_path),
        "output_dir": str(output_dir),
        "command": cmd,
        "start_time": time.time(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V5 across multiple GPUs, generating each step prompt online with the trained Qwen planner."
    )
    parser.add_argument(
        "--ids_file",
        "--evaluate_ids",
        "--evaluate_ids_file",
        dest="ids_file",
        type=str,
        default=None,
        help=(
            "Optional text file containing comma-separated or newline-separated video IDs. "
            "If omitted, IDs are read from --manifest_path."
        ),
    )
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--step_source", choices=["qwen_planner", "qwen_open_loop", "manifest"], default="qwen_planner",
                        help="qwen_planner predicts each next action; qwen_open_loop predicts the full plan once; manifest replays reference captions.")
    parser.add_argument("--prompt_manifest_path", type=str, default=None,
                        help="Caption source for --step_source manifest only.")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--initial_preview_clip", type=str, default=None)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Defaults to planner_ for Qwen or Puzzle_ for manifest captions.")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3")
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help=(
            "0-based inclusive start index into manifest/evaluate_ids list. "
            "Use with --end_index to shard evaluation across nodes."
        ),
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help=(
            "0-based exclusive end index into manifest/evaluate_ids list. "
            "Omit to run through the end."
        ),
    )
    parser.add_argument("--master_port_base", type=int, default=29500)
    parser.add_argument("--poll_interval_sec", type=float, default=5.0)
    parser.add_argument(
        "--max_parallel_jobs",
        type=int,
        default=None,
        help=(
            "Maximum worker processes to keep active at once. "
            "Use this to avoid host RAM SIGKILLs while loading large checkpoints."
        ),
    )
    parser.add_argument(
        "--launch_stagger_sec",
        type=float,
        default=60.0,
        help="Seconds to wait between worker launches.",
    )
    parser.add_argument(
        "--resume",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Skip videos that already have both step_prompt_summary.json and final output video.",
    )
    parser.add_argument(
        "--skip_existing",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="Deprecated alias for --resume.",
    )
    parser.add_argument("--dry_run", type=str_to_bool, nargs="?", const=True, default=False)

    parser.add_argument("--action_ckpt", type=str, default=_default_action_ckpt())
    parser.add_argument("--model_path", type=str, default=_default_model_path())
    parser.add_argument("--final_output_name", type=str, default="gen.mp4")
    parser.add_argument("--save_summary_json", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    parser.add_argument("--aspect_ratio", type=str, default="16:9")
    parser.add_argument("--video_length", type=int, default=None)
    parser.add_argument("--target_duration_seconds", type=float, default=30.0,
                        help="Exact output duration for --step_source qwen_open_loop.")
    parser.add_argument("--step_video_length", type=int, default=77)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--flow_shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=3208)
    parser.add_argument("--chunk_latent_frames", type=int, default=4)
    parser.add_argument("--reference_frame_mode", type=str, default="last", choices=["first", "last"])
    parser.add_argument(
        "--reference_source",
        type=str,
        default=None,
        choices=[
            "generated_prev",
            "manifest_gold_every_step",
            "manifest_reference_clip_first_every_step",
            "train_aligned_auto",
        ],
        help=(
            "Defaults to generated_prev for Qwen planning, train_aligned_auto for manifest captions."
        ),
    )
    parser.add_argument(
        "--continuity_mode",
        type=str,
        default="stateless_prompt_ref",
        choices=["stateless_prompt_ref", "latent_memory", "last"],
        help=(
            "Continuity strategy. Use stateless_prompt_ref for per-step generation. "
            "'last' is accepted as a shortcut for stateless_prompt_ref with --reference_frame_mode last."
        ),
    )
    parser.add_argument("--prompt_memory_steps", type=int, default=0)
    parser.add_argument(
        "--seed_schedule",
        type=str,
        default="increment",
        choices=["fixed", "increment"],
    )
    parser.add_argument("--memory_frames", type=int, default=20)
    parser.add_argument("--temporal_context_size", type=int, default=12)
    parser.add_argument(
        "--memory_frame_policy",
        type=str,
        default="recent",
        choices=["recent", "aligned"],
    )
    parser.add_argument("--origami_memory_mode", type=str, default=os.getenv("ORIGAMI_MEMORY_MODE", "mode2"))
    parser.add_argument("--origami_memory_steps", type=int, default=int(os.getenv("ORIGAMI_MEMORY_STEPS", "2")))
    parser.add_argument("--origami_memory_policy", type=str, default=os.getenv("ORIGAMI_MEMORY_POLICY", "latest_k"))
    parser.add_argument("--origami_memory_blend", type=float, default=float(os.getenv("ORIGAMI_MEMORY_BLEND", "1.0")))
    parser.add_argument("--model_type", type=str, default="ar", choices=["ar", "bi"])
    parser.add_argument("--save_step_latents", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--stitch_drop_first_after_step1", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--output_fps", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--offloading", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--group_offloading", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--enable_torch_compile", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_sageattn", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--sage_blocks_range", type=str, default="0-53")
    parser.add_argument("--use_vae_parallel", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_fp8_gemm", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--quant_type", type=str, default="fp8-per-block")
    parser.add_argument("--include_patterns", type=str, default="double_blocks")
    parser.add_argument("--few_step", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "--transformer_resident_ar_rollout",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    return add_planner_arguments(parser)


def normalize_args(args) -> None:
    if args.output_prefix is None:
        planner_prefix = "planner_" if args.planner_visual_feedback else "planner_no_visual_"
        args.output_prefix = {"qwen_planner": planner_prefix, "qwen_open_loop": "open_loop_", "manifest": "Puzzle_"}[args.step_source]
    if args.reference_source is None:
        args.reference_source = "train_aligned_auto" if args.step_source == "manifest" else "generated_prev"

    def _largest_compatible_chunk(latent_length: int, requested_chunk: int) -> int:
        requested = max(1, min(requested_chunk, latent_length))
        for candidate in range(requested, 0, -1):
            if latent_length % candidate == 0:
                return candidate
        return 1

    if args.continuity_mode == "last":
        args.continuity_mode = "stateless_prompt_ref"
        args.reference_frame_mode = "last"
    if args.step_video_length <= 0:
        return
    latent_length = ((args.step_video_length - 1) // 4) + 1
    if args.model_type == "bi":
        # Training denoises the complete target window in one transformer call.
        args.chunk_latent_frames = latent_length
    else:
        args.chunk_latent_frames = _largest_compatible_chunk(
            latent_length, args.chunk_latent_frames
        )
    if (
        args.memory_frame_policy == "aligned"
        and args.temporal_context_size > args.memory_frames
    ):
        args.temporal_context_size = args.memory_frames


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    normalize_args(args)
    if args.step_source in ("qwen_planner", "qwen_open_loop"):
        validate_planner_args(args)
    expected_frame_count = None
    if args.step_source == "qwen_open_loop":
        if args.prompt_memory_steps or not args.planner_visual_feedback:
            raise ValueError("Open-loop planning requires prompt_memory_steps=0 and planner_visual_feedback=true (initial image only).")
        if args.video_length is not None:
            raise ValueError("Use --target_duration_seconds for open-loop generation, not --video_length.")
        schedule = frame_schedule(args.target_duration_seconds, args.output_fps, args.step_video_length,
                                  args.stitch_drop_first_after_step1)
        if len(schedule) > args.max_planned_steps:
            raise ValueError("The duration requires more chunks than --max_planned_steps.")
        expected_frame_count = sum(schedule)
    if args.ids_file and not os.path.exists(args.ids_file):
        raise FileNotFoundError(f"Missing ids_file: {args.ids_file}")
    if not os.path.exists(args.manifest_path):
        raise FileNotFoundError(f"Missing manifest_path: {args.manifest_path}")
    if args.prompt_manifest_path and not os.path.exists(args.prompt_manifest_path):
        raise FileNotFoundError(f"Missing prompt_manifest_path: {args.prompt_manifest_path}")
    if args.image_path and not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Missing image_path: {args.image_path}")
    if args.initial_preview_clip and not os.path.exists(args.initial_preview_clip):
        raise FileNotFoundError(f"Missing initial_preview_clip: {args.initial_preview_clip}")
    if args.master_port_base <= 0:
        raise ValueError("master_port_base must be positive.")
    if args.poll_interval_sec <= 0:
        raise ValueError("poll_interval_sec must be positive.")
    if args.max_parallel_jobs is not None and args.max_parallel_jobs <= 0:
        raise ValueError("max_parallel_jobs must be positive when provided.")
    if args.launch_stagger_sec < 0:
        raise ValueError("launch_stagger_sec must be >= 0.")
    if args.start_index < 0:
        raise ValueError("start_index must be >= 0.")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("end_index must be >= start_index.")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    max_parallel_jobs = min(args.max_parallel_jobs or len(gpu_ids), len(gpu_ids))
    step_latent_length = ((args.step_video_length - 1) // 4) + 1
    video_ids = select_video_ids(
        args.manifest_path,
        args.ids_file,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing is not None:
        args.resume = args.skip_existing

    selected_ids: list[str] = []
    skipped_completed: list[str] = []
    for video_id in video_ids:
        output_dir = output_root / f"{args.output_prefix}{sanitize_filename(video_id)}"
        if should_skip_video(
            output_dir, args.final_output_name, args.resume, args.step_source, args.planner_visual_feedback,
            expected_frame_count,
            args.output_fps if args.step_source == "qwen_open_loop" else None,
        ):
            skipped_completed.append(video_id)
            continue
        selected_ids.append(video_id)

    print(
        f"[plan] step_source={args.step_source} ids_total={len(video_ids)} selected={len(selected_ids)} "
        f"planner_visual_feedback={args.planner_visual_feedback} "
        f"skipped_completed={len(skipped_completed)} resume={args.resume} "
        f"index_range=[{args.start_index},{args.end_index if args.end_index is not None else 'end'}) "
        f"gpus={gpu_ids} max_parallel_jobs={max_parallel_jobs} "
        f"launch_stagger_sec={args.launch_stagger_sec} "
        f"step_video_length={args.step_video_length} latent_length={step_latent_length} "
        f"chunk_latent_frames={args.chunk_latent_frames} "
        f"model_type={args.model_type} "
        f"memory_frames={args.memory_frames} temporal_context_size={args.temporal_context_size} "
        f"memory_frame_policy={args.memory_frame_policy} "
        f"master_ports={[args.master_port_base + i for i in range(len(gpu_ids))]}"
    )
    print(
        f"[origami-memory] mode={args.origami_memory_mode}, steps={args.origami_memory_steps}, "
        f"policy={args.origami_memory_policy}, blend={args.origami_memory_blend}"
    )
    if skipped_completed:
        print(
            f"[resume] first_selected={selected_ids[0] if selected_ids else '<none>'} "
            f"last_skipped_completed={skipped_completed[-1]}"
        )
    if args.dry_run:
        for idx, video_id in enumerate(selected_ids):
            slot_index = idx % len(gpu_ids)
            gpu_id = gpu_ids[slot_index]
            output_dir = output_root / f"{args.output_prefix}{sanitize_filename(video_id)}"
            cmd = build_generation_command(
                args,
                video_id=video_id,
                output_dir=output_dir,
                master_port=args.master_port_base + slot_index,
            )
            print(f"[dry-run] gpu={gpu_id} port={args.master_port_base + slot_index} video_id={video_id}")
            print("          " + " ".join(cmd))
        return

    if selected_ids:
        checked = check_inference_checkpoints(args)
        print(f"[preflight] python={sys.executable} checked_indexed_shards={checked}", flush=True)

    pending = list(selected_ids)
    active_by_slot: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    successes: list[str] = []

    try:
        while pending or active_by_slot:
            for slot_index, gpu_id in enumerate(gpu_ids):
                if len(active_by_slot) >= max_parallel_jobs:
                    break
                if slot_index in active_by_slot:
                    continue
                if not pending:
                    break
                video_id = pending.pop(0)
                job = launch_job(args, video_id, gpu_id, slot_index, logs_dir)
                active_by_slot[slot_index] = job
                print(
                    f"[launch] gpu={gpu_id} slot={slot_index} "
                    f"hip_visible={gpu_id} port={job['master_port']} "
                    f"video_id={video_id} log={job['log_path']}"
                )
                if (
                    args.launch_stagger_sec > 0
                    and pending
                    and len(active_by_slot) < max_parallel_jobs
                ):
                    time.sleep(args.launch_stagger_sec)

            finished_slots: list[int] = []
            for slot_index, job in active_by_slot.items():
                return_code = job["process"].poll()
                if return_code is None:
                    continue
                duration = time.time() - job["start_time"]
                job["log_handle"].close()
                if return_code == 0:
                    successes.append(job["video_id"])
                    print(f"[done] video_id={job['video_id']} gpu={job['gpu_id']} duration={duration:.1f}s")
                else:
                    failures.append(
                        {
                            "video_id": job["video_id"],
                            "gpu_id": job["gpu_id"],
                            "return_code": return_code,
                            "log_path": job["log_path"],
                            "command": job["command"],
                        }
                    )
                    print(
                        f"[fail] video_id={job['video_id']} gpu={job['gpu_id']} "
                        f"rc={return_code} log={job['log_path']}"
                    )
                finished_slots.append(slot_index)

            for slot_index in finished_slots:
                active_by_slot.pop(slot_index, None)

            if pending or active_by_slot:
                time.sleep(args.poll_interval_sec)
    finally:
        for job in active_by_slot.values():
            try:
                job["process"].terminate()
            except Exception:
                pass
            try:
                job["log_handle"].close()
            except Exception:
                pass

    print(
        f"[summary] succeeded={len(successes)} failed={len(failures)} total_selected={len(selected_ids)}"
    )
    if failures:
        raise RuntimeError(
            "Some jobs failed. "
            + "; ".join(
                f"{item['video_id']} (gpu={item['gpu_id']}, rc={item['return_code']}, log={item['log_path']})"
                for item in failures
            )
        )


if __name__ == "__main__":
    main()

"""

  --evaluate_ids_file assets/gold_test_ids.txt \


cd worldGuide
export PYTHONPATH=$(pwd):$PYTHONPATH

PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 hyvideo/run_step_prompts_v5_multi_gpu.py \
  --step_source manifest \
  --manifest_path data/train_manifest.json \
  --prompt_manifest_path data/train_manifest.json \
  --output_root ./outputs/worldguide_v5_try1_guidance7_5_chunkLatentFrames12_new \
  --start_index 204 \
  --end_index 231 \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --max_parallel_jobs 8 \
  --launch_stagger_sec 0 \
  --resume true \
  --model_path ckpts/HunyuanVideo-1.5 \
  --action_ckpt ckpts/worldguide_action/diffusion_pytorch_model.safetensors \
  --step_video_length 45 \
  --num_inference_steps 50 \
  --guidance_scale 7.5 \
  --flow_shift 5.0 \
  --reference_source generated_prev \
  --continuity_mode stateless_prompt_ref \
  --reference_frame_mode last \
  --chunk_latent_frames 12 \
  --memory_frames 20 \
  --temporal_context_size 12 \
  --memory_frame_policy recent \
  --origami_memory_mode mode2 \
  --origami_memory_steps 2 \
  --origami_memory_policy latest_k \
  --origami_memory_blend 1.0 \
  --resolution 480p \
  --aspect_ratio 16:9 \
  --height 480 \
  --width 832 \
  --output_fps 16 \
  --dtype bf16 \
  --seed 28

"""

# PYTORCH_ALLOC_CONF=expandable_segments:True python3 hyvideo/run_step_prompts_v5_multi_gpu.py   --manifest_path data/test_manifest.json   --output_root ./outputs/worldguide_step_caption_v5_testdata_retry2   --gpu_ids 0,1,2,3   --max_parallel_jobs 4   --launch_stagger_sec 0   --master_port_base 29500   --model_path ckpts/HunyuanVideo-1.5   --action_ckpt ckpts/worldguide_action/diffusion_pytorch_model.safetensors  --step_video_length 33   --num_inference_steps 30   --guidance_scale 6.0   --flow_shift 5.0   --reference_source generated_prev   --continuity_mode stateless_prompt_ref   --reference_frame_mode last   --chunk_latent_frames 9   --memory_frames 20   --temporal_context_size 12   --memory_frame_policy recent   --resolution 480p   --aspect_ratio 16:9   --height 480   --width 832   --output_fps 16   --dtype bf16   --seed 3208


# 0: 0–33
# 1: 33–74
# 2: 74–107
# 3: 107–143
# 4: 143–179
# 5: 179–204
# 6: 204–231
# 7: 231–263
# PYTORCH_ALLOC_CONF=expandable_segments:True \
# python3 hyvideo/run_step_prompts_v5_multi_gpu.py \
#   --step_source manifest \
#   --manifest_path data/test_manifest.json \
#   --prompt_manifest_path data/train_manifest.json \
#   --output_root ./outputs/worldguide_step_caption_v5_testdata_retry2 \
#   --gpu_ids 0,1,2,3,4,5,6,7 \
#   --max_parallel_jobs 8 \
#   --launch_stagger_sec 0 \
#   --resume true \
#   --model_path ckpts/HunyuanVideo-1.5 \
#   --action_ckpt ckpts/worldguide_action/diffusion_pytorch_model.safetensors \
#   --step_video_length 33 \
#   --num_inference_steps 30 \
#   --guidance_scale 6.0 \
#   --flow_shift 5.0 \
#   --reference_source generated_prev \
#   --continuity_mode stateless_prompt_ref \
#   --reference_frame_mode last \
#   --chunk_latent_frames 9 \
#   --memory_frames 4 \
#   --temporal_context_size 12 \
#   --memory_frame_policy recent \
#   --origami_memory_mode mode2 \
#   --origami_memory_steps 2 \
#   --origami_memory_policy latest_k \
#   --origami_memory_blend 0.0 \
#   --resolution 480p \
#   --aspect_ratio 16:9 \
#   --height 480 \
#   --width 832 \
#   --output_fps 16 \
#   --dtype bf16 \
#   --seed 3208
