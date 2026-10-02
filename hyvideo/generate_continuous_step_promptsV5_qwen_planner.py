"""Closed-loop V5 generation where every step prompt comes from the Qwen planner.

Same generation path as generate_continuous_step_promptsV5.py (reference image,
mode2 memory latents, seeds, stitching, summary), but the manifest caption plan
is NOT used.  From the manifest row we only take the goal and the starting
observation (task / query / category / work_subject / global_clip_path /
initial_frame_path).  Each step then runs:

    planner(goal, last generated clip, last K predicted captions) -> caption
    HunyuanVideo(caption, reference, memory)                     -> clip

Planner inputs mirror YUME_VLA/fastvideo/dataset/vla_dataset.py (training):
  * system prompt: PLANNER_SYSTEM_PROMPT
  * task goal:     "A step-by-step {category} tutorial showing how to make {work_subject}."
  * step 1:        global_clip_path preview video (nframes=8) + env-preview prompt
  * step N >= 2:   last generated clip only (nframes=8) + last K predicted captions
Generation stops when the planner emits <|Task Completed|> or the step cap is hit.
With --planner_visual_feedback false, Qwen receives only the goal and predicted
action history. Hunyuan's visual reference and memory inputs remain enabled.
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

import imageio.v2 as imageio
import torch
import torch.distributed as dist
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from hyvideo import generate_continuous_step_promptsV5 as v5
from hyvideo.inference_checks import write_json_atomic
from hyvideo.generate_continuous_step_promptsV5 import (
    TASK_COMPLETED_TOKEN,
    cleanup_tensors,
    is_rank0,
    rank0_log,
)
from hyvideo.qwen_planner_utils import (
    ACTION_HISTORY_PLANNER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    add_planner_arguments,
    build_planner_prompt,
    build_task_goal,
    clean_planner_prediction,
    split_completion,
    validate_planner_args,
)


class QwenStepPlanner:
    def __init__(self, args) -> None:
        self.args = args
        from hyvideo.qwen_video_reader import configure_planner_video_backend
        configure_planner_video_backend(args.planner_video_backend)
        self.dtype = torch.bfloat16 if args.planner_dtype == "bf16" else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor_path = args.planner_processor_path or args.planner_model_path
        rank0_log(f"[planner] loading Qwen planner from {args.planner_model_path}")
        self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        if TASK_COMPLETED_TOKEN not in self.processor.tokenizer.get_vocab():
            self.processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": [TASK_COMPLETED_TOKEN]}
            )
        config = AutoConfig.from_pretrained(args.planner_model_path, trust_remote_code=True)
        if isinstance(getattr(config, "text_config", None), dict):
            # Saved by transformers>=4.52 (nested text_config, duplicated at top level).
            # Older transformers keep it as a raw dict and crash in GenerationConfig.
            delattr(config, "text_config")
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.planner_model_path,
            config=config,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self._on_device = False
        if not args.planner_offload:
            self._to_device()

    def _to_device(self) -> None:
        if not self._on_device:
            self.model.to(self.device)
            self._on_device = True

    def _to_cpu(self) -> None:
        if self._on_device:
            self.model.to("cpu")
            self._on_device = False
            cleanup_tensors()

    def visual_part(self, visual: dict[str, str]) -> dict[str, Any]:
        if visual["type"] == "video":
            return {
                "type": "video",
                "video": visual["path"],
                "nframes": self.args.planner_nframes,
                "max_pixels": self.args.planner_max_pixels,
                "min_pixels": self.args.planner_min_pixels,
            }
        return {
            "type": "image", "image": visual["path"],
            "max_pixels": self.args.planner_max_pixels,
            "min_pixels": self.args.planner_min_pixels,
        }

    @torch.no_grad()
    def plan(self, task_goal: str, step_number: int, visual: dict[str, str] | None, history: list[str]) -> tuple[str, str, str]:
        visual_feedback = self.args.planner_visual_feedback
        if visual_feedback and visual is None:
            raise ValueError("Visual-feedback planning requires a current observation.")
        prompt = build_planner_prompt(
            task_goal=task_goal,
            history=history,
            step_number=step_number,
            is_env_preview=visual_feedback and step_number == 1 and visual["type"] == "video",
            max_history_text=self.args.planner_history_k,
            visual_feedback=visual_feedback,
        )
        content = [{"type": "text", "text": prompt}]
        if visual_feedback:
            content.insert(0, self.visual_part(visual))
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT if visual_feedback else ACTION_HISTORY_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        self.last_plan_attempts = []
        raw = self.generate_messages(messages, use_visual_inputs=visual_feedback)
        prediction = clean_planner_prediction(raw, self.processor)
        self.last_plan_attempts.append({
            "planner_raw_output": raw, "planner_output": prediction, "decoding": "greedy",
        })
        if not prediction and self.args.planner_retry_empty:
            rank0_log(f"[planner-retry] step={step_number} empty raw={raw!r}; retrying once with initial transport tokens suppressed")
            # Same goal, observation and history. Let Qwen choose an action or
            # its trained completion token; never supply a fallback caption.
            raw = self.generate_messages(
                messages, use_visual_inputs=visual_feedback, suppress_initial_special_tokens=True,
            )
            prediction = clean_planner_prediction(raw, self.processor)
            self.last_plan_attempts.append({
                "planner_raw_output": raw, "planner_output": prediction,
                "decoding": "greedy_initial_transport_tokens_suppressed",
            })
        return prompt, raw, prediction

    @torch.no_grad()
    def generate_messages(self, messages: list[dict[str, Any]], use_visual_inputs: bool,
                          *, full_plan_actions: int | None = None,
                          suppress_initial_special_tokens: bool = False) -> str:
        """Run one model.generate call for either a next action or a full plan."""
        if full_plan_actions is not None and suppress_initial_special_tokens:
            raise ValueError("Empty next-action retry cannot be combined with full-plan decoding.")
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = video_inputs = None
        vision_kwargs = {}
        if use_visual_inputs:
            image_inputs, video_inputs = process_vision_info(messages)
            vision_kwargs = {"images": image_inputs, "videos": video_inputs}
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            **vision_kwargs,
        )
        # The processor has finished consuming decoded frames. Do not keep the
        # full-resolution inputs alive while the planner is on the GPU.
        del image_inputs, video_inputs, vision_kwargs
        constraints = None
        generation_options = {}
        self.last_generation_constraints = None
        if suppress_initial_special_tokens:
            tokenizer = self.processor.tokenizer
            completion_id = tokenizer.get_vocab().get(TASK_COMPLETED_TOKEN)
            if completion_id is None:
                raise ValueError("Empty planner retry requires the trained Task Completed token.")
            blocked_ids = sorted(set(tokenizer.all_special_ids) - {completion_id})
            if not blocked_ids:
                raise ValueError("Empty planner retry found no transport tokens to suppress.")
            # Transformers applies this mask only to the first new token.
            # EOS remains available afterward; Task Completed is never masked.
            generation_options["begin_suppress_tokens"] = blocked_ids
        if full_plan_actions is not None:
            from hyvideo.open_loop_logits_processor import FullPlanLogitsProcessor
            constraints = FullPlanLogitsProcessor(
                self.processor.tokenizer, full_plan_actions, self.args.planner_max_new_tokens,
            )
            generation_options = {
                "logits_processor": [constraints], "num_beams": 1,
                "eos_token_id": constraints.eos_token_id, "no_repeat_ngram_size": 8,
            }
        output_ids = trimmed = None
        try:
            self._to_device()
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.args.planner_max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                **generation_options,
            )
            trimmed = output_ids[0][inputs["input_ids"].shape[1]:]
            self.last_generation_token_count = trimmed.numel()
            raw = self.processor.decode(trimmed, skip_special_tokens=False).strip()
            if constraints is not None:
                self.last_generation_constraints = constraints.metadata()
        finally:
            del inputs, output_ids, trimmed
            if self.args.planner_offload:
                self._to_cpu()
            else:
                cleanup_tensors()
        return raw


def broadcast_from_rank0(value: Any) -> Any:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        payload = [value]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]
    return value


def load_goal_row(args) -> dict[str, Any]:
    rows = [row for row in v5.load_json_rows(args.manifest_path) if row.get("video_id") == args.video_id]
    if not rows:
        raise ValueError(f"No rows found for video_id='{args.video_id}' in manifest: {args.manifest_path}")
    # Per-step manifests: goal fields are shared, step-1 row carries global_clip_path.
    rows.sort(key=lambda row: int(row.get("step_number") or 0))
    return rows[0]


def resolve_step1_planner_visual(row: dict[str, Any]) -> dict[str, str]:
    global_clip_path = row.get("global_clip_path")
    if isinstance(global_clip_path, str) and os.path.exists(global_clip_path):
        return {"type": "video", "path": global_clip_path}
    initial_frame_path = row.get("initial_frame_path")
    if isinstance(initial_frame_path, str) and os.path.exists(initial_frame_path):
        rank0_log(
            "[planner] WARNING global_clip_path missing; step 1 planner sees initial_frame_path "
            "as an image, which differs from training (1s preview video)."
        )
        return {"type": "image", "path": initial_frame_path}
    raise FileNotFoundError(
        f"video_id={row.get('video_id')}: neither global_clip_path nor initial_frame_path exists."
    )


def initial_observation_row(row: dict[str, Any], args) -> dict[str, Any]:
    """Resolve one starting state for both models, excluding per-step gold data."""
    if args.initial_preview_clip or args.image_path:
        return {
            "video_id": row.get("video_id"),
            "global_clip_path": args.initial_preview_clip,
            "initial_frame_path": args.image_path,
        }
    return {key: row.get(key) for key in ("video_id", "global_clip_path", "initial_frame_path")}


def log_ablation_memory(stage: str, step_number: int | None = None) -> None:
    """Report live memory at ablation boundaries without changing the pipeline."""
    if os.getenv("HYVIDEO_DEBUG_STAGES", "0") != "1":
        return
    fields = [f"[ablation-memory] stage={stage} step={step_number}"]
    try:
        status = Path("/proc/self/status").read_text()
        rss_kib = next(line.split()[1] for line in status.splitlines() if line.startswith("VmRSS:"))
        fields.append(f"host_rss_gib={int(rss_kib) / 1024**2:.3f}")
    except (OSError, StopIteration, ValueError):
        pass
    if torch.cuda.is_available():
        fields.extend([
            f"gpu_allocated_gib={torch.cuda.memory_allocated() / 1024**3:.3f}",
            f"gpu_reserved_gib={torch.cuda.memory_reserved() / 1024**3:.3f}",
            f"gpu_peak_allocated_gib={torch.cuda.max_memory_allocated() / 1024**3:.3f}",
        ])
    rank0_log(" ".join(fields))


def run_planner_step_generation(pipe, planner: QwenStepPlanner | None, output_root: Path, args) -> dict[str, Any]:
    clips_dir = output_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_root / args.final_output_name

    row = load_goal_row(args)
    task_goal = build_task_goal(row)
    observation_row = initial_observation_row(row, args)
    initial_visual = resolve_step1_planner_visual(observation_row)
    visual_feedback = args.planner_visual_feedback
    max_steps = args.video_length if args.video_length is not None else args.max_planned_steps
    step_latent_length = v5.latent_length_from_video_length(args.step_video_length)
    rank0_log(f"[planner] video_id={args.video_id} task_goal={task_goal!r} max_steps={max_steps}")

    generated_clip_paths: list[str] = []
    step_results: list[dict[str, Any]] = []
    planner_trace: list[dict[str, Any]] = []
    planner_history: list[str] = []
    prompt_history: list[str] = []
    step_seeds: list[int] = []
    # RGB clips live on disk. Keep only mode2's required latest-K states, on
    # CPU, and transfer the selected latent window back for the next clip.
    history_size = args.origami_memory_steps if args.origami_memory_mode == "mode2" else 0
    generated_step_latents = deque(maxlen=history_size)
    generated_reference_images = deque(maxlen=history_size)
    cumulative_generated_frames = 0
    stop_reason = "plan_limit_reached"

    def save_progress(stage: str, step_number: int | None = None, error: str | None = None) -> None:
        if is_rank0():
            write_json_atomic(output_root / "planner_progress.json", {
                "video_id": args.video_id, "step_source": "qwen_planner", "stage": stage,
                "step_number": step_number, "error": error,
                "planner_model_path": args.planner_model_path,
                "planner_video_backend": args.planner_video_backend,
                "generated_step_count": len(step_results), "planner_trace": planner_trace,
                "steps": step_results,
            })

    for step_idx in range(max_steps):
        step_number = step_idx + 1
        is_first_step = step_idx == 0

        # 1) Plan the next action from the current (generated) state.
        if not visual_feedback:
            planner_visual = None
        elif is_first_step:
            planner_visual = initial_visual
        else:
            planner_visual = {"type": "video", "path": generated_clip_paths[-1]}
        planned = None
        save_progress("planning", step_number)
        log_ablation_memory("before_planning", step_number)
        if is_rank0():
            try:
                planned = {"prediction": planner.plan(
                    task_goal=task_goal,
                    step_number=step_number,
                    visual=planner_visual,
                    history=planner_history,
                ), "attempts": planner.last_plan_attempts}
            except Exception as exc:
                # All ranks must leave the loop together if planning fails.
                planned = {"error": f"{type(exc).__name__}: {exc}"}
        planned = broadcast_from_rank0(planned)
        if "error" in planned:
            save_progress("planner_failed", step_number, planned["error"])
            raise RuntimeError(f"Planner failed at step {step_number}: {planned['error']}")
        planner_prompt, planner_raw, planner_output = planned["prediction"]
        action, completed = split_completion(planner_output)
        rank0_log(f"[planner] step={step_number} raw={planner_raw!r} action={action!r} completed={completed}")
        planner_trace.append({
            "step_number": step_number,
            "planner_prompt": planner_prompt,
            "planner_raw_output": planner_raw,
            "planner_output": planner_output,
            "planner_visual_input": planner_visual,
            "planner_emitted_task_completed": completed,
            "planner_attempts": planned["attempts"],
        })
        save_progress("planned", step_number)

        if not action:
            if not completed:
                error = f"Planner produced an empty action at step {step_number} (raw={planner_raw!r})."
                save_progress("planner_failed", step_number, error)
                raise RuntimeError(error)
            stop_reason = "task_completed"
            break

        # 2) Generate the clip exactly like V5 stateless_prompt_ref / generated_prev.
        if generated_clip_paths:
            reference_image, reference_visual_path, reference_source_type = (
                v5.build_later_step_inputs_with_mode(generated_clip_paths, args.reference_frame_mode)
            )
        else:
            reference_image, reference_visual_path, reference_source_type = (
                v5.build_user_initial_inputs(args, observation_row)
            )

        step_prompt, step_text = v5.normalize_step_prompt(
            action, expected_step_number=step_number
        )
        if args.prompt_memory_steps > 0:
            memory_steps = prompt_history[-args.prompt_memory_steps:]
            if memory_steps:
                memory_prefix = "Previous steps context:\n" + "\n".join(f"- {txt}" for txt in memory_steps)
                step_prompt = f"{memory_prefix}\n\nCurrent step:\n{step_prompt}"

        step_seed = args.seed + step_idx if args.seed_schedule == "increment" else args.seed
        step_seeds.append(step_seed)

        memory_history_latents = None
        if is_first_step:
            memory_history_latents = v5.build_global_clip_memory_latents(
                pipe=pipe,
                global_clip_path=str(observation_row.get("global_clip_path") or ""),
                latent_length=step_latent_length,
                args=args,
            )
        elif generated_step_latents and args.origami_memory_mode == "mode2":
            if args.origami_memory_policy != "latest_k":
                raise ValueError(
                    f"Unsupported origami_memory_policy={args.origami_memory_policy!r}; expected 'latest_k'."
                )
            memory_history_latents = v5.pad_or_trim_memory_latents(
                torch.cat(list(generated_step_latents), dim=2), max(step_latent_length - 1, 0)
            ).to(pipe.execution_device) * args.origami_memory_blend

        clip_output_path = clips_dir / f"step_{step_number:02d}.mp4"
        save_progress("generating_clip", step_number)
        log_ablation_memory("before_generation", step_number)
        generation_result = v5.generate_step_clip(
            pipe=pipe,
            step_prompt=step_prompt,
            reference_image=reference_image,
            video_length=args.step_video_length,
            output_path=clip_output_path,
            step_seed=step_seed,
            args=args,
            memory_history_latents=memory_history_latents,
            memory_reference_images=list(generated_reference_images),
        )
        generated_latents = generation_result.pop("_generated_latents")
        if history_size:
            # copy=True also releases a larger backing allocation when the
            # returned latents are a view. Preserve the original values/dtype;
            # re-encoding the saved lossy MP4 would change the memory ablation.
            generated_step_latents.append(generated_latents.detach().to("cpu", copy=True))
            generated_reference_images.append(reference_image.copy())
        del generated_latents, memory_history_latents, reference_image
        # generate_step_clip already releases decoded frames after saving.
        # Its returned GPU latents and this step's memory window are now gone.
        cleanup_tensors()
        log_ablation_memory("clip_saved_and_released", step_number)
        generated_clip_paths.append(str(clip_output_path))
        # Feed back only executed actions, without control tokens or labels.
        planner_history.append(step_text)
        prompt_history.append(step_text)
        cumulative_generated_frames += args.step_video_length

        step_results.append(
            {
                "step_number": step_number,
                "task_goal": task_goal,
                "planner_prompt": planner_prompt,
                "planner_raw_output": planner_raw,
                "planner_output": planner_output,
                "planner_visual_input": planner_visual,
                "planner_emitted_task_completed": completed,
                "raw_step_prompt": action,
                "step_prompt": step_prompt,
                "step_text": step_text,
                "clip_path": str(clip_output_path),
                "reference_visual_path": reference_visual_path,
                "initial_visual_source_type": reference_source_type,
                "manifest_initial_frame_path": row.get("initial_frame_path"),
                "manifest_global_clip_path": row.get("global_clip_path"),
                "generated_frames_this_step": args.step_video_length,
                "cumulative_generated_frames_after_step": cumulative_generated_frames,
                "step_seed": step_seed,
                "continuity_mode": args.continuity_mode,
                "stitch_drop_first_after_step1": args.stitch_drop_first_after_step1,
                **generation_result,
            }
        )
        save_progress("clip_saved", step_number)
        rank0_log(
            f"Generated step {step_number}/{max_steps} | step_prompt={step_prompt} | seed={step_seed} | "
            f"reference_source={reference_source_type} | cumulative_generated_frames={cumulative_generated_frames}"
        )

        if completed and args.stop_on_task_completed:
            stop_reason = "task_completed"
            break

    final_written_frames = None
    generated_step_latents.clear()
    generated_reference_images.clear()
    cleanup_tensors()
    save_progress("stitching")
    log_ablation_memory("before_stitching")
    if is_rank0():
        writer = imageio.get_writer(final_output_path, fps=args.output_fps)
        written_frames = 0
        try:
            stitch_paths = generated_clip_paths
            if not stitch_paths:
                # A token-only completion at step 1 means the starting state is
                # already finished. Save that observation without inventing an action.
                if initial_visual["type"] == "video":
                    stitch_paths = [initial_visual["path"]]
                else:
                    writer.append_data(imageio.imread(initial_visual["path"]))
                    written_frames += 1
            for clip_idx, clip_path in enumerate(stitch_paths):
                reader = imageio.get_reader(clip_path)
                try:
                    for frame_idx, frame in enumerate(reader):
                        if args.stitch_drop_first_after_step1 and clip_idx > 0 and frame_idx == 0:
                            continue
                        writer.append_data(frame)
                        written_frames += 1
                finally:
                    reader.close()
        finally:
            writer.close()
        final_written_frames = written_frames
        rank0_log(f"Saved final video to {final_output_path} | written_frames={written_frames}")

    summary = {
        "manifest_path": args.manifest_path,
        "video_id": args.video_id,
        "step_source": "qwen_planner",
        "planner_model_path": args.planner_model_path,
        "planner_visual_feedback": visual_feedback,
        "planner_system_prompt": PLANNER_SYSTEM_PROMPT if visual_feedback else ACTION_HISTORY_PLANNER_SYSTEM_PROMPT,
        "planner_history_k": args.planner_history_k,
        "planner_nframes": args.planner_nframes,
        "planner_processor_path": args.planner_processor_path or args.planner_model_path,
        "planner_max_pixels": args.planner_max_pixels,
        "planner_min_pixels": args.planner_min_pixels,
        "planner_video_backend": args.planner_video_backend,
        "planner_max_new_tokens": args.planner_max_new_tokens,
        "planner_retry_empty": args.planner_retry_empty,
        "vae_decode_offload_transformer": args.vae_decode_offload_transformer,
        "stop_on_task_completed": args.stop_on_task_completed,
        "planner_trace": planner_trace,
        "initial_planner_visual": initial_visual if visual_feedback else None,
        "task_goal": task_goal,
        "manifest_task": row.get("task"),
        "manifest_query": row.get("query"),
        "manifest_work_subject": row.get("work_subject"),
        # Ground truth kept only for offline scoring; never shown to the planner.
        "reference_full_caption": row.get("full_caption"),
        "reference_total_steps": row.get("total_steps"),
        "reference_source": args.reference_source,
        "action_ckpt": args.action_ckpt,
        "strict_checkpoint_load": True,
        "max_planned_steps": max_steps,
        "generated_step_count": len(step_results),
        "stop_reason": stop_reason,
        "step_video_length": args.step_video_length,
        "total_generated_frames": cumulative_generated_frames,
        "final_written_frames": final_written_frames,
        "output_fps": args.output_fps,
        "final_output_path": str(final_output_path),
        "continuity": {
            "mode": args.continuity_mode,
            "seed_schedule": args.seed_schedule,
            "step_seeds": step_seeds,
            "prompt_memory_steps": args.prompt_memory_steps,
            "memory_frames": args.memory_frames,
            "temporal_context_size": args.temporal_context_size,
            "memory_frame_policy": args.memory_frame_policy,
            "reference_frame_mode": args.reference_frame_mode,
            "origami_memory_mode": args.origami_memory_mode,
            "origami_memory_steps": args.origami_memory_steps,
            "origami_memory_policy": args.origami_memory_policy,
            "origami_memory_blend": args.origami_memory_blend,
            "stitch_drop_first_after_step1": args.stitch_drop_first_after_step1,
        },
        "steps": step_results,
    }
    if is_rank0() and args.save_summary_json:
        summary_path = output_root / "step_prompt_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        rank0_log(f"Saved step-prompt summary to {summary_path}")
    save_progress("complete")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = add_planner_arguments(v5.build_parser())
    parser.set_defaults(reference_source="generated_prev")
    parser.description = "Qwen-planner-driven step generation for one manifest video_id (V5 generation path)."
    return parser


def main() -> None:
    args = build_parser().parse_args(v5.normalize_leading_dash_arg_values(sys.argv[1:]))

    validate_planner_args(args)
    v5.validate_initial_inputs(args)
    row = load_goal_row(args)
    build_task_goal(row)
    resolve_step1_planner_visual(initial_observation_row(row, args))

    v5.initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    v5.initialize_infer_state(args)

    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    rank0_log(
        f"Starting Qwen-planner generation | video_id={args.video_id} | planner={args.planner_model_path} | "
        f"planner_visual_feedback={args.planner_visual_feedback} | "
        f"action_ckpt={args.action_ckpt} | step_video_length={args.step_video_length} | model_type={args.model_type}"
    )
    rank0_log(f"[runtime] python={sys.executable} torch={torch.__version__} "
              f"hip={torch.version.hip} planner_video_backend={args.planner_video_backend} "
              f"offloading={args.offloading} group_offloading={args.group_offloading} "
              f"planner_offload={args.planner_offload} "
              f"vae_decode_offload_transformer={args.vae_decode_offload_transformer} "
              f"planner_retry_empty={args.planner_retry_empty}")
    rank0_log(
        f"[runtime-config] PYTORCH_ALLOC_CONF={os.getenv('PYTORCH_ALLOC_CONF', '<unset>')} "
        f"cpu_threads={torch.get_num_threads()} "
        f"AMD_SERIALIZE_KERNEL={os.getenv('AMD_SERIALIZE_KERNEL', '<unset>')} "
        f"AMD_SERIALIZE_COPY={os.getenv('AMD_SERIALIZE_COPY', '<unset>')}"
    )
    if args.origami_memory_mode == "mode2" and args.model_type != "bi":
        rank0_log("[origami-memory] WARNING: mode2 with model_type=ar discards memory tokens; use --model_type bi.")

    pipe = None
    planner = None
    try:
        pipe = v5.create_pipeline(args)
        if args.vae_decode_offload_transformer:
            from hyvideo.ablation_vae_memory import install_vae_decode_transformer_offload
            install_vae_decode_transformer_offload(pipe, log=rank0_log)
        planner = QwenStepPlanner(args) if is_rank0() else None
        run_planner_step_generation(pipe=pipe, planner=planner, output_root=output_root, args=args)
        v5.stage_barrier()
    finally:
        del pipe
        del planner
        cleanup_tensors(finalize_distributed=True)


if __name__ == "__main__":
    main()
