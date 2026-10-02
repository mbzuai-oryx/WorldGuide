"""Predict one full action plan from an initial image, then render bounded clips.

Qwen runs exactly once before Hunyuan is loaded. Its validated plan is saved
before rendering. No generated state is ever returned to the planner.
"""

import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import torch
from PIL import Image

from hyvideo import generate_continuous_step_promptsV5 as v5
from hyvideo.generate_continuous_step_promptsV5_qwen_planner import (
    QwenStepPlanner, broadcast_from_rank0, initial_observation_row, load_goal_row,
)
from hyvideo.open_loop_planner_utils import (
    OPEN_LOOP_SYSTEM_PROMPT, build_full_plan_prompt, frame_schedule, parse_full_plan,
)
from hyvideo.qwen_planner_utils import add_planner_arguments, build_task_goal, validate_planner_args


def resolve_initial_image(row: dict[str, Any], args, output_root: Path) -> str:
    observation = initial_observation_row(row, args)
    image_path = observation.get("initial_frame_path")
    if image_path and Path(image_path).is_file():
        return str(image_path)
    preview_path = observation.get("global_clip_path")
    if preview_path and Path(preview_path).is_file():
        # The planner receives one image, never a preview video.
        path = output_root / "planner_initial_frame.png"
        if v5.is_rank0():
            v5.load_first_frame_image(preview_path).save(path)
        v5.stage_barrier()
        return str(path)
    raise FileNotFoundError("Open-loop planning needs initial_frame_path / --image_path, or a preview to extract its first frame.")


def predict_full_plan(planner: QwenStepPlanner, row: dict[str, Any], initial_image: str,
                      output_root: Path, args) -> dict[str, Any]:
    schedule = frame_schedule(args.target_duration_seconds, args.output_fps,
                              args.step_video_length, args.stitch_drop_first_after_step1)
    goal = build_task_goal(row)
    prompt = build_full_plan_prompt(goal, len(schedule), args.target_duration_seconds)
    messages = [
        {"role": "system", "content": OPEN_LOOP_SYSTEM_PROMPT},
        {"role": "user", "content": [
            planner.visual_part({"type": "image", "path": initial_image}),
            {"type": "text", "text": prompt},
        ]},
    ]
    raw = planner.generate_messages(messages, use_visual_inputs=True, full_plan_actions=len(schedule))
    plan = {
        "video_id": args.video_id,
        "planning_mode": "open_loop",
        "planner_call_count": 1,
        "planner_model_path": args.planner_model_path,
        "planner_system_prompt": OPEN_LOOP_SYSTEM_PROMPT,
        "planner_prompt": prompt,
        "planner_raw_output": raw,
        "planner_generated_tokens": planner.last_generation_token_count,
        "planner_max_new_tokens": args.planner_max_new_tokens,
        "planner_decoding": planner.last_generation_constraints,
        "planner_visual_input": {"type": "image", "path": initial_image},
        "task_goal": goal,
        "target_duration_seconds": args.target_duration_seconds,
        "output_fps": args.output_fps,
        "target_frame_count": sum(schedule),
        "retained_frames_per_clip": schedule,
        "requested_action_count": len(schedule),
        "status": "invalid",
    }
    try:
        plan["actions"] = parse_full_plan(raw, planner.processor, len(schedule))
        plan["status"] = "ready"
    except ValueError as exc:
        plan["error"] = str(exc)
        raise
    finally:
        # Keep even a malformed prediction for inspection; never fill it with gold actions.
        if v5.is_rank0():
            (output_root / "open_loop_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if v5.is_rank0():
        (output_root / "planned_actions.txt").write_text(
            "\n".join(f"Step {i}: {action}" for i, action in enumerate(plan["actions"], 1)) + "\n",
            encoding="utf-8",
        )
        v5.rank0_log(f"[open-loop] saved all {len(plan['actions'])} actions from one Qwen call")
    return plan


def stitch_clips(clip_paths: list[str], keep_counts: list[int], output_path: Path, args) -> int:
    if not keep_counts or any(keep <= 0 for keep in keep_counts) or len(clip_paths) != len(keep_counts):
        raise ValueError("The number of generated clips does not match the fixed plan.")
    written = 0
    temporary_path = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    writer = imageio.get_writer(temporary_path, fps=args.output_fps)
    try:
        for index, (clip_path, keep) in enumerate(zip(clip_paths, keep_counts)):
            reader = imageio.get_reader(clip_path)
            retained = 0
            try:
                for frame_index, frame in enumerate(reader):
                    if args.stitch_drop_first_after_step1 and index > 0 and frame_index == 0:
                        continue
                    writer.append_data(frame)
                    written += 1
                    retained += 1
                    if retained == keep:
                        break
            finally:
                reader.close()
            if retained != keep:
                raise RuntimeError(f"Clip {clip_path} has only {retained} usable frames; expected {keep}.")
    except BaseException:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
    if written != sum(keep_counts):
        raise RuntimeError(f"Expected {sum(keep_counts)} output frames, wrote {written}.")
    temporary_path.replace(output_path)
    return written


def generate_fixed_plan(pipe, plan: dict[str, Any], row: dict[str, Any], output_root: Path, args) -> dict[str, Any]:
    """Render the saved actions in order. This function has no planner/model handle."""
    clips_dir = output_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    # The model consumes only the most recent K states. Bound device memory too.
    latents = deque(maxlen=max(args.origami_memory_steps, 1))
    references = deque(maxlen=max(args.origami_memory_steps, 1))
    steps = []
    schedule = plan["retained_frames_per_clip"]
    if len(plan["actions"]) != len(schedule):
        raise ValueError("The saved action sequence does not match the frame schedule.")
    step_latent_length = v5.latent_length_from_video_length(args.step_video_length)
    observation = initial_observation_row(row, args)
    output_start = 0
    for index, action in enumerate(plan["actions"]):
        step_number = index + 1
        if not clips:
            reference_path = plan["planner_visual_input"]["path"]
            reference = Image.open(reference_path).convert("RGB")
            reference_source = "initial_image"
        else:
            reference, reference_path, reference_source = v5.build_later_step_inputs_with_mode(clips, args.reference_frame_mode)
        memory = None
        if index == 0:
            memory = v5.build_global_clip_memory_latents(
                pipe=pipe, global_clip_path=str(observation.get("global_clip_path") or ""),
                latent_length=step_latent_length, args=args,
            )
        elif args.origami_memory_mode == "mode2":
            memory = v5.pad_or_trim_memory_latents(torch.cat(list(latents), dim=2), step_latent_length - 1) * args.origami_memory_blend
        prompt, step_text = v5.normalize_step_prompt(action, expected_step_number=step_number)
        seed = args.seed + index if args.seed_schedule == "increment" else args.seed
        path = clips_dir / f"step_{step_number:02d}.mp4"
        result = v5.generate_step_clip(
            pipe=pipe, step_prompt=prompt, reference_image=reference,
            video_length=args.step_video_length, output_path=path, step_seed=seed,
            args=args, memory_history_latents=memory, memory_reference_images=list(references),
        )
        latents.append(result.pop("_generated_latents"))
        references.append(reference.copy())
        clips.append(str(path))
        steps.append({
            "step_number": step_number, "step_prompt": prompt, "step_text": step_text,
            "planner_output": action, "clip_path": str(path), "step_seed": seed,
            "reference_visual_path": reference_path, "initial_visual_source_type": reference_source,
            "retained_frames": schedule[index], "output_start_frame": output_start,
            "output_end_frame_exclusive": output_start + schedule[index], **result,
        })
        output_start += schedule[index]
        v5.rank0_log(f"[open-loop] saved chunk {step_number}/{len(schedule)}: {action}")
    final_path = output_root / args.final_output_name
    written = stitch_clips(clips, schedule, final_path, args) if v5.is_rank0() else None
    summary = {
        "video_id": args.video_id, "manifest_path": args.manifest_path,
        "step_source": "qwen_open_loop", "planning_mode": "open_loop", "planner_call_count": 1,
        "planner_model_path": args.planner_model_path, "planner_system_prompt": OPEN_LOOP_SYSTEM_PROMPT,
        "planner_trace": [{key: plan[key] for key in ("planner_prompt", "planner_raw_output", "planner_visual_input")}],
        "open_loop_plan_path": str(output_root / "open_loop_plan.json"),
        "task_goal": plan["task_goal"], "action_ckpt": args.action_ckpt,
        "step_video_length": args.step_video_length, "generated_step_count": len(steps),
        "total_generated_frames": len(steps) * args.step_video_length,
        "target_duration_seconds": args.target_duration_seconds, "target_frame_count": sum(schedule),
        "final_written_frames": written, "output_fps": args.output_fps,
        "final_output_path": str(final_path), "stop_reason": "plan_executed",
        "continuity": {key: getattr(args, key) for key in (
            "model_type", "reference_source", "reference_frame_mode", "origami_memory_mode",
            "origami_memory_steps", "origami_memory_policy", "origami_memory_blend",
            "stitch_drop_first_after_step1", "seed_schedule",
        )},
        "steps": steps,
    }
    if v5.is_rank0() and args.save_summary_json:
        (output_root / "step_prompt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser():
    parser = add_planner_arguments(v5.build_parser())
    parser.description = "One Qwen full-plan prediction from the initial image, followed by fixed-duration chunk generation."
    parser.set_defaults(planner_max_new_tokens=2048)
    parser.add_argument("--target_duration_seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args(v5.normalize_leading_dash_arg_values(sys.argv[1:]))
    validate_planner_args(args)
    v5.validate_initial_inputs(args)
    if args.video_length is not None:
        raise ValueError("Use --target_duration_seconds for open-loop evaluation, not --video_length.")
    schedule = frame_schedule(args.target_duration_seconds, args.output_fps,
                              args.step_video_length, args.stitch_drop_first_after_step1)
    if len(schedule) > args.max_planned_steps:
        raise ValueError("The duration requires more chunks than --max_planned_steps.")
    if args.prompt_memory_steps:
        raise ValueError("Open-loop generation uses its fixed action sequence; set --prompt_memory_steps 0.")
    if not args.planner_visual_feedback:
        raise ValueError("Open-loop planning requires the initial image; keep --planner_visual_feedback true.")
    row = load_goal_row(args)
    build_task_goal(row)
    v5.initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    v5.initialize_infer_state(args)
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    planner = pipe = None
    try:
        if v5.is_rank0():
            # An interrupted rerun must not inherit a previous successful summary.
            (output_root / "step_prompt_summary.json").write_text(json.dumps({
                "video_id": args.video_id, "step_source": "qwen_open_loop", "stop_reason": "planning",
            }), encoding="utf-8")
        initial_image = resolve_initial_image(row, args, output_root)
        payload = None
        if v5.is_rank0():
            try:
                planner = QwenStepPlanner(args)
                payload = {"plan": predict_full_plan(planner, row, initial_image, output_root, args)}
            except Exception as exc:
                payload = {"error": f"{type(exc).__name__}: {exc}"}
        payload = broadcast_from_rank0(payload)
        if "error" in payload:
            raise RuntimeError(f"Full-plan prediction failed: {payload['error']}")
        del planner
        planner = None
        v5.cleanup_tensors()
        pipe = v5.create_pipeline(args)
        generate_fixed_plan(pipe, payload["plan"], row, output_root, args)
        v5.stage_barrier()
    finally:
        del planner, pipe
        v5.cleanup_tensors(finalize_distributed=True)


if __name__ == "__main__":
    main()
