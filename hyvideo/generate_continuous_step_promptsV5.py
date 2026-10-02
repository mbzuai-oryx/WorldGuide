import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

if (
    "PYTORCH_ALLOC_CONF" not in os.environ
    and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

import imageio.v2 as imageio
import loguru
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from PIL import Image

from hyvideo.commons import auto_offload_model
from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from trainer.dataset.origami_step_dataset import _build_static_intrinsic, _build_static_w2c
from trainer.dataset.transform import CenterCropResizeVideo

STEP_TEXT_KEYS = ("target_caption", "step_text", "step_prompt", "caption")
CONTINUITY_MODES = ("stateless_prompt_ref", "latent_memory")
SEED_SCHEDULES = ("fixed", "increment")
TASK_COMPLETED_TOKEN = "<|Task Completed|>"
STEP_RE = re.compile(r"(?<!\w)Step\s*(\d{1,3})\s*:\s*", re.IGNORECASE)


def _default_model_path() -> str:
    return os.environ.get("MODEL_PATH", "ckpts/HunyuanVideo-1.5")


def _default_action_ckpt() -> str:
    return os.environ.get("ACTION_CKPT", "ckpts/worldguide_action/diffusion_pytorch_model.safetensors")


def is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def rank0_log(message: str) -> None:
    if is_rank0():
        loguru.logger.info(message)


def stage_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


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


def normalize_leading_dash_arg_values(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    idx = 0
    while idx < len(argv):
        current = argv[idx]
        if current == "--video_id" and idx + 1 < len(argv):
            normalized.append(f"--video_id={argv[idx + 1]}")
            idx += 2
            continue
        normalized.append(current)
        idx += 1
    return normalized


def cleanup_tensors(finalize_distributed: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    if finalize_distributed and dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def save_video(video: torch.Tensor, path: Path, fps: int) -> None:
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    video = (video * 255).clamp(0, 255).to(torch.uint8)
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    imageio.mimwrite(path, video, fps=fps)


def _load_frame_image_with_imageio(video_path: str, frame_mode: str) -> Image.Image:
    reader = imageio.get_reader(video_path)
    try:
        if frame_mode == "first":
            frame = reader.get_data(0)
        elif frame_mode == "last":
            frame = None
            for candidate in reader:
                frame = candidate
            if frame is None:
                raise ValueError(f"No frames found in video: {video_path}")
        else:
            raise ValueError(
                f"Unsupported frame_mode: {frame_mode}. Expected 'first' or 'last'."
            )
    finally:
        reader.close()

    if frame.ndim == 2:
        return Image.fromarray(frame, mode="L").convert("RGB")
    if frame.ndim == 3 and frame.shape[2] > 3:
        frame = frame[:, :, :3]
    return Image.fromarray(frame).convert("RGB")


def _load_video_frames_with_imageio(video_path: str) -> tuple[torch.Tensor, float]:
    """Decode a whole video to a uint8 TCHW tensor plus its source fps."""
    reader = imageio.get_reader(video_path)
    try:
        try:
            source_fps = float(reader.get_meta_data().get("fps", 0.0) or 0.0)
        except Exception:
            source_fps = 0.0
        frames = []
        for frame in reader:
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[2] > 3:
                frame = frame[:, :, :3]
            frames.append(frame)
    finally:
        reader.close()

    if not frames:
        return torch.empty(0, 3, 1, 1, dtype=torch.uint8), source_fps
    stacked = torch.from_numpy(np.stack(frames, axis=0))
    return stacked.permute(0, 3, 1, 2).contiguous(), source_fps


def load_first_frame_image(video_path: str) -> Image.Image:
    read_video = getattr(torchvision.io, "read_video", None)
    if callable(read_video):
        frames, _, _ = read_video(video_path, output_format="TCHW")
        if frames.shape[0] == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        first_frame = frames[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(first_frame)
    rank0_log(
        "torchvision.io.read_video is unavailable; falling back to imageio for first-frame loading."
    )
    return _load_frame_image_with_imageio(video_path, frame_mode="first")


def load_last_frame_image(video_path: str) -> Image.Image:
    read_video = getattr(torchvision.io, "read_video", None)
    if callable(read_video):
        frames, _, _ = read_video(video_path, output_format="TCHW")
        if frames.shape[0] == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        last_frame = frames[-1].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(last_frame)
    rank0_log(
        "torchvision.io.read_video is unavailable; falling back to imageio for last-frame loading."
    )
    return _load_frame_image_with_imageio(video_path, frame_mode="last")


def load_reference_frame_image(video_path: str, frame_mode: str) -> Image.Image:
    if frame_mode == "first":
        return load_first_frame_image(video_path)
    if frame_mode == "last":
        return load_last_frame_image(video_path)
    raise ValueError(
        f"Unsupported reference_frame_mode: {frame_mode}. Expected 'first' or 'last'."
    )


def sample_frame_indices(total_frames: int, source_fps: float, target_fps: int) -> torch.Tensor:
    if total_frames <= 0:
        return torch.empty(0, dtype=torch.long)
    if not source_fps or source_fps <= 0 or target_fps <= 0:
        return torch.arange(total_frames, dtype=torch.long)

    frame_interval = source_fps / float(target_fps)
    if frame_interval <= 1.0:
        return torch.arange(total_frames, dtype=torch.long)

    indices = torch.arange(0, total_frames, frame_interval, dtype=torch.float32).long()
    return indices.clamp(0, total_frames - 1)


def load_video_tensor_for_vae(
    video_path: str, height: int, width: int, target_fps: int
) -> torch.Tensor:
    read_video = getattr(torchvision.io, "read_video", None)
    if callable(read_video):
        frames, _, metadata = read_video(video_path, output_format="TCHW")
        source_fps = float(metadata.get("video_fps", 0.0) or 0.0)
    else:
        frames, source_fps = _load_video_frames_with_imageio(video_path)
    if frames.shape[0] == 0:
        raise ValueError(f"No frames found in global clip: {video_path}")
    sample_indices = sample_frame_indices(frames.shape[0], source_fps, target_fps)
    frames = frames[sample_indices]
    frames = CenterCropResizeVideo((height, width))(frames)
    return frames.permute(1, 0, 2, 3).float() / 127.5 - 1.0


def pad_or_trim_memory_latents(memory_latents: torch.Tensor, target_frames: int) -> torch.Tensor:
    if memory_latents.shape[2] > target_frames:
        return memory_latents[:, :, -target_frames:, :, :]
    if memory_latents.shape[2] < target_frames:
        pad_frames = target_frames - memory_latents.shape[2]
        pad = memory_latents.new_zeros(
            memory_latents.shape[0],
            memory_latents.shape[1],
            pad_frames,
            memory_latents.shape[3],
            memory_latents.shape[4],
        )
        return torch.cat([pad, memory_latents], dim=2)
    return memory_latents


def build_global_clip_memory_latents(
    pipe: HunyuanVideo_1_5_Pipeline,
    global_clip_path: str,
    latent_length: int,
    args,
) -> torch.Tensor | None:
    if args.origami_memory_mode != "mode2" or not global_clip_path:
        return None
    if not os.path.exists(global_clip_path):
        if is_rank0():
            rank0_log(f"[origami-memory] global_clip_path missing, step 1 uses reference image only: {global_clip_path}")
        return None

    target_hist_frames = max(latent_length - 1, 0)
    if target_hist_frames <= 0:
        return None

    device = pipe.execution_device
    try:
        video_tensor = load_video_tensor_for_vae(
            global_clip_path,
            height=args.height,
            width=args.width,
            target_fps=args.output_fps,
        ).unsqueeze(0).to(device)
        with auto_offload_model(
            pipe.vae, pipe.execution_device, enabled=pipe.enable_offloading
        ):
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                memory_hist = pipe.vae.encode(video_tensor).latent_dist.mode()
                memory_hist = memory_hist * pipe.vae.config.scaling_factor
        memory_hist = pad_or_trim_memory_latents(memory_hist, target_hist_frames)
        return memory_hist * args.origami_memory_blend
    except Exception as exc:
        if is_rank0():
            rank0_log(
                "[origami-memory] failed to encode global_clip_path for step 1; "
                f"using reference image only. path={global_clip_path} error={exc}"
            )
        return None


def build_static_controls(latent_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    viewmats = _build_static_w2c(latent_length).unsqueeze(0).cpu()
    intrinsics = _build_static_intrinsic(latent_length).unsqueeze(0).cpu()
    action = torch.zeros((1, latent_length), dtype=torch.long)
    return viewmats, intrinsics, action


def latent_length_from_video_length(video_length: int) -> int:
    if video_length <= 0:
        raise ValueError(f"video_length must be positive, got {video_length}")
    return ((video_length - 1) // 4) + 1


def largest_compatible_chunk_size(latent_length: int, requested_chunk: int) -> int:
    requested = max(1, min(requested_chunk, latent_length))
    for candidate in range(requested, 0, -1):
        if latent_length % candidate == 0:
            return candidate
    return 1


def load_json_rows(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        if isinstance(payload, dict):
            for key in ("records", "data", "items", "videos", "samples"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [payload]
        raise ValueError(f"Manifest is not a list or object: {path}")
    return payload


def _extract_step_text(row: dict[str, Any]) -> str | None:
    for key in STEP_TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def parse_full_caption_steps(full_caption: Any) -> list[dict[str, Any]]:
    text = "" if full_caption is None else str(full_caption)
    text = text.replace(TASK_COMPLETED_TOKEN, " ")
    text = re.sub(r"<\|Task Completed\|>", " ", text, flags=re.IGNORECASE).strip()
    if not text:
        return []

    matches = list(STEP_RE.finditer(text))
    if not matches:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        return [
            {
                "source_step_number": 1,
                "step_number": 1,
                "caption": cleaned,
            }
        ]

    steps: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        caption = re.sub(r"\s+", " ", text[start:end].strip(" \t\r\n;")).strip()
        if not caption:
            continue
        steps.append(
            {
                "source_step_number": int(match.group(1)),
                "step_number": len(steps) + 1,
                "caption": caption,
            }
        )
    return steps


def build_full_caption_step_plan(
    args, row: dict[str, Any], prompt_manifest_path: str
) -> tuple[list[dict[str, Any]], str]:
    parsed_steps = parse_full_caption_steps(row.get("full_caption"))
    if not parsed_steps:
        raise ValueError(
            f"No Step N captions found in full_caption for video_id={args.video_id}."
        )

    max_steps = (
        len(parsed_steps)
        if args.video_length is None
        else min(args.video_length, len(parsed_steps))
    )
    selected_steps = parsed_steps[:max_steps]
    stop_reason = (
        "full_caption_exhausted"
        if len(parsed_steps) <= max_steps
        else "plan_limit_reached"
    )

    step_plan: list[dict[str, Any]] = []
    for expected_step_number, parsed_step in enumerate(selected_steps, start=1):
        step_text = parsed_step["caption"]
        source_step_number = parsed_step.get("source_step_number")
        prompt_step_number = expected_step_number
        if (
            isinstance(source_step_number, int)
            and source_step_number > 0
            and source_step_number != expected_step_number
        ):
            rank0_log(
                f"Full caption label mismatch for video_id={args.video_id}: "
                f"parsed Step {source_step_number}, using sequential Step {expected_step_number}."
            )

        step_prompt, normalized_step_text = normalize_step_prompt(
            f"Step {prompt_step_number}: {step_text}",
            expected_step_number=expected_step_number,
            label_step_number=prompt_step_number,
        )
        prompt_source_row = {
            **row,
            "step_number": expected_step_number,
            "source_step_number": source_step_number,
            "caption": step_text,
            "target_caption": step_text,
            "step_text": step_text,
        }
        step_plan.append(
            {
                "step_number": expected_step_number,
                "prompt_step_number": prompt_step_number,
                "manifest_row": row,
                "prompt_source_row": prompt_source_row,
                "raw_step_prompt": step_text,
                "step_prompt": step_prompt,
                "step_text": normalized_step_text,
                "prompt_manifest_path": prompt_manifest_path,
            }
        )
    return step_plan, stop_reason


def _index_manifest_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    by_sample_id: dict[str, dict[str, Any]] = {}
    by_video_step: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        video_id = row.get("video_id")
        step_number = row.get("step_number")
        if isinstance(sample_id, str) and sample_id:
            by_sample_id[sample_id] = row
        if isinstance(video_id, str) and isinstance(step_number, int):
            by_video_step[(video_id, step_number)] = row
    return by_sample_id, by_video_step


def candidate_prompt_manifest_paths(
    primary_manifest_path: str, explicit_prompt_manifest_path: str | None
) -> list[str]:
    if explicit_prompt_manifest_path:
        return [explicit_prompt_manifest_path]

    primary_dir = str(Path(primary_manifest_path).resolve().parent)
    candidates = [primary_manifest_path]
    if primary_manifest_path.endswith("_frames.json"):
        candidates.append(primary_manifest_path.replace("_frames.json", ".json"))
    candidates.append(os.path.join(primary_dir, "train_manifest_upto_1000007.json"))
    candidates.append(os.path.join(primary_dir, "train_manifest_final_sample200.json"))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def load_manifest_step_plan(args) -> tuple[list[dict[str, Any]], str]:
    primary_rows = load_json_rows(args.manifest_path)
    video_rows = [row for row in primary_rows if row.get("video_id") == args.video_id]
    if not video_rows:
        raise ValueError(
            f"No rows found for video_id='{args.video_id}' in manifest: {args.manifest_path}"
        )

    full_caption_rows = [
        row
        for row in video_rows
        if isinstance(row.get("full_caption"), str) and row.get("full_caption", "").strip()
    ]
    has_per_step_rows = any(isinstance(row.get("step_number"), int) for row in video_rows)
    if full_caption_rows and not has_per_step_rows:
        if len(full_caption_rows) > 1:
            raise ValueError(
                f"Found {len(full_caption_rows)} full_caption rows for video_id={args.video_id}; "
                "expected one row per video."
            )
        return build_full_caption_step_plan(
            args,
            row=full_caption_rows[0],
            prompt_manifest_path=args.manifest_path,
        )

    try:
        video_rows.sort(key=lambda row: int(row.get("step_number", 0)))
    except Exception as exc:
        raise ValueError("Could not sort rows by step_number.") from exc

    max_steps = len(video_rows) if args.video_length is None else min(args.video_length, len(video_rows))
    selected_rows = video_rows[:max_steps]
    stop_reason = "manifest_exhausted" if len(video_rows) <= max_steps else "plan_limit_reached"

    prompt_manifest_path = ""
    by_sample_id: dict[str, dict[str, Any]] = {}
    by_video_step: dict[tuple[str, int], dict[str, Any]] = {}
    best_coverage = -1
    for candidate in candidate_prompt_manifest_paths(
        args.manifest_path, args.prompt_manifest_path
    ):
        if not os.path.exists(candidate):
            continue
        candidate_rows = load_json_rows(candidate)
        candidate_by_sample_id, candidate_by_video_step = _index_manifest_rows(
            candidate_rows
        )
        covered_steps = 0
        for row in selected_rows:
            sample_id = row.get("sample_id")
            raw_step_number = row.get("step_number")
            sample_row = (
                candidate_by_sample_id.get(sample_id)
                if isinstance(sample_id, str)
                else None
            )
            if sample_row is not None and _extract_step_text(sample_row) is not None:
                covered_steps += 1
                continue
            if isinstance(raw_step_number, int):
                step_row = candidate_by_video_step.get((args.video_id, raw_step_number))
                if step_row is not None and _extract_step_text(step_row) is not None:
                    covered_steps += 1
                    continue
        if covered_steps > best_coverage:
            best_coverage = covered_steps
            prompt_manifest_path = candidate
            by_sample_id = candidate_by_sample_id
            by_video_step = candidate_by_video_step
        if covered_steps == len(selected_rows):
            break

    step_plan: list[dict[str, Any]] = []
    for expected_step_number, row in enumerate(selected_rows, start=1):
        raw_step_number = row.get("step_number")
        if isinstance(raw_step_number, int) and raw_step_number != expected_step_number:
            rank0_log(
                f"Step sequence gap for video_id={args.video_id}: expected {expected_step_number}, "
                f"manifest row has step_number={raw_step_number}. Proceeding with sorted order."
            )

        prompt_source_row = row
        step_text = _extract_step_text(prompt_source_row)
        if step_text is None:
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str) and sample_id in by_sample_id:
                prompt_source_row = by_sample_id[sample_id]
                step_text = _extract_step_text(prompt_source_row)
            if step_text is None and isinstance(raw_step_number, int):
                key = (args.video_id, raw_step_number)
                if key in by_video_step:
                    prompt_source_row = by_video_step[key]
                    step_text = _extract_step_text(prompt_source_row)

        if step_text is None:
            raise ValueError(
                "Could not find per-step text prompt for "
                f"video_id={args.video_id}, step={expected_step_number}. "
                "Provide --prompt_manifest_path that contains one of: "
                f"{', '.join(STEP_TEXT_KEYS)}. "
                f"Current selected prompt manifest: {prompt_manifest_path}"
            )

        prompt_step_number = (
            raw_step_number
            if isinstance(raw_step_number, int) and raw_step_number > 0
            else expected_step_number
        )
        step_prompt, normalized_step_text = normalize_step_prompt(
            f"Step {prompt_step_number}: {step_text}",
            expected_step_number=expected_step_number,
            label_step_number=prompt_step_number,
        )
        step_plan.append(
            {
                "step_number": expected_step_number,
                "prompt_step_number": prompt_step_number,
                "manifest_row": row,
                "prompt_source_row": prompt_source_row,
                "raw_step_prompt": step_text,
                "step_prompt": step_prompt,
                "step_text": normalized_step_text,
                "prompt_manifest_path": prompt_manifest_path,
            }
        )

    return step_plan, stop_reason


def normalize_step_prompt(
    raw_step: str, expected_step_number: int, label_step_number: int | None = None
) -> tuple[str, str]:
    match = re.match(r"^\s*Step\s+(\d+)\s*:\s*(.+?)\s*$", raw_step, flags=re.IGNORECASE)
    final_label = (
        label_step_number
        if isinstance(label_step_number, int) and label_step_number > 0
        else expected_step_number
    )
    if match:
        step_label = int(match.group(1))
        step_text = match.group(2).strip()
        if not step_text:
            raise ValueError(f"Missing step text in prompt: {raw_step}")
        if step_label != expected_step_number:
            rank0_log(
                f"Step file label mismatch: expected Step {expected_step_number}, "
                f"but found Step {step_label}. Using file contents as written."
            )
        return f"Step {final_label}: {step_text}", step_text

    step_text = raw_step.strip()
    if not step_text:
        raise ValueError(f"Empty step prompt at step {expected_step_number}")

    return f"Step {final_label}: {step_text}", step_text



def create_pipeline(args) -> HunyuanVideo_1_5_Pipeline:
    from hyvideo.inference_checks import check_inference_checkpoints
    check_inference_checkpoints(args)
    transformer_version = f"{args.resolution}_i2v"
    if args.dtype == "bf16":
        transformer_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    return HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=transformer_dtype,
        action_ckpt=args.action_ckpt,
    )


def generate_step_clip(
    pipe: HunyuanVideo_1_5_Pipeline,
    step_prompt: str,
    reference_image: Image.Image,
    video_length: int,
    output_path: Path,
    step_seed: int,
    args,
    memory_history_latents: torch.Tensor | None = None,
    memory_reference_images: list[Image.Image] | None = None,
) -> dict[str, Any]:
    generation_output = None
    decoded = None
    latent_length = latent_length_from_video_length(video_length)
    try:
        viewmats, intrinsics, action = build_static_controls(latent_length)
        chunk_latent_frames = args.chunk_latent_frames
        if chunk_latent_frames is None:
            chunk_latent_frames = 4
        if chunk_latent_frames <= 0:
            raise ValueError("chunk_latent_frames must be positive.")
        requested_chunk_latent_frames = chunk_latent_frames
        if args.model_type == "bi":
            # Training denoises the whole window in one forward pass. Forcing a
            # single chunk reproduces that; any smaller chunk would re-introduce
            # cross-chunk context handling that training never saw.
            chunk_latent_frames = latent_length
            if chunk_latent_frames != requested_chunk_latent_frames:
                rank0_log(
                    "model_type=bi denoises the full window in one pass; "
                    f"using chunk_latent_frames={chunk_latent_frames} instead of "
                    f"requested {requested_chunk_latent_frames}."
                )
        else:
            chunk_latent_frames = largest_compatible_chunk_size(
                latent_length, chunk_latent_frames
            )
            if chunk_latent_frames != requested_chunk_latent_frames:
                rank0_log(
                    "chunk_latent_frames is incompatible with latent_length "
                    f"({requested_chunk_latent_frames} vs {latent_length}); "
                    f"using compatible chunk_latent_frames={chunk_latent_frames}."
                )
        generation_output = pipe(
            prompt=step_prompt,
            aspect_ratio=args.aspect_ratio,
            video_length=video_length,
            prompt_rewrite=False,
            num_inference_steps=args.num_inference_steps,
            negative_prompt=args.negative_prompt,
            seed=step_seed,
            flow_shift=args.flow_shift,
            embedded_guidance_scale=args.embedded_guidance_scale,
            reference_image=reference_image,
            user_height=args.height,
            user_width=args.width,
            chunk_latent_frames=chunk_latent_frames,
            guidance_scale=args.guidance_scale,
            viewmats=viewmats,
            Ks=intrinsics,
            action=action,
            enable_sr=False,
            output_type="pt",
            return_dict=True,
            few_step=args.few_step,
            model_type=args.model_type,
            transformer_resident_ar_rollout=args.transformer_resident_ar_rollout,
            memory_frames=args.memory_frames,
            temporal_context_size=args.temporal_context_size,
            memory_frame_policy=args.memory_frame_policy,
            origami_memory_mode=args.origami_memory_mode,
            origami_memory_steps=args.origami_memory_steps,
            origami_memory_policy=args.origami_memory_policy,
            origami_memory_blend=args.origami_memory_blend,
            memory_history_latents=memory_history_latents,
            memory_reference_images=memory_reference_images,
        )
        decoded = generation_output.videos
        if decoded is None:
            raise RuntimeError(f"Pipeline returned empty video for prompt: {step_prompt}")
        generated_latents = generation_output.latents
        if generated_latents is None:
            raise RuntimeError(f"Pipeline returned empty latents for prompt: {step_prompt}")
        if os.getenv("ORIGAMI_DEBUG", "0") == "1":
            decoded_has_nan = bool(torch.isnan(decoded).any().item())
            latents_has_nan = bool(torch.isnan(generated_latents).any().item())
            rank0_log(
                f"[memory-debug] step_output decoded_has_nan={decoded_has_nan} "
                f"latents_has_nan={latents_has_nan}"
            )
        save_error_path = f"{output_path}.save_error"
        save_exception_message = None
        if is_rank0():
            try:
                rank0_log(f"[clip-save] start path={output_path}")
                save_video(decoded, output_path, fps=args.output_fps)
                rank0_log(f"[clip-save] done path={output_path}")
                if os.path.exists(save_error_path):
                    os.unlink(save_error_path)
            except Exception as exc:
                save_exception_message = str(exc)
                with open(save_error_path, "w", encoding="utf-8") as fp:
                    fp.write(save_exception_message)
        stage_barrier()
        if os.path.exists(save_error_path):
            with open(save_error_path, "r", encoding="utf-8") as fp:
                save_error_text = fp.read().strip()
            raise RuntimeError(
                f"Failed saving step clip at {output_path}: {save_error_text}"
            )
        if save_exception_message is not None:
            raise RuntimeError(
                f"Failed saving step clip at {output_path}: {save_exception_message}"
            )
        return {
            "video_length": video_length,
            "latent_length": latent_length,
            "chunk_latent_frames": chunk_latent_frames,
            "output_path": str(output_path),
            "_generated_latents": generated_latents.detach(),
        }
    finally:
        del generation_output
        del decoded
        cleanup_tensors()


def validate_initial_inputs(args) -> None:
    if not os.path.exists(args.manifest_path):
        raise FileNotFoundError(f"Missing manifest_path: {args.manifest_path}")
    if args.prompt_manifest_path and not os.path.exists(args.prompt_manifest_path):
        raise FileNotFoundError(f"Missing prompt_manifest_path: {args.prompt_manifest_path}")
    if not args.video_id:
        raise ValueError("--video_id is required.")
    if args.image_path and not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Missing image_path: {args.image_path}")
    if args.initial_preview_clip and not os.path.exists(args.initial_preview_clip):
        raise FileNotFoundError(
            f"Missing initial_preview_clip: {args.initial_preview_clip}"
        )


def build_initial_inputs_from_manifest_row(
    args, manifest_row: dict[str, Any]
) -> tuple[Image.Image, str, str]:
    initial_frame_path = manifest_row.get("initial_frame_path")
    if (
        isinstance(initial_frame_path, str)
        and initial_frame_path
        and os.path.exists(initial_frame_path)
    ):
        reference_image = Image.open(initial_frame_path).convert("RGB")
        return reference_image, initial_frame_path, "manifest_initial_frame"

    gold_frame_path = manifest_row.get("gold_frame_path")
    if isinstance(gold_frame_path, str) and gold_frame_path and os.path.exists(gold_frame_path):
        reference_image = Image.open(gold_frame_path).convert("RGB")
        return reference_image, gold_frame_path, "manifest_gold_frame"

    if args.image_path:
        reference_image = Image.open(args.image_path).convert("RGB")
        return reference_image, args.image_path, "user_image"

    if args.initial_preview_clip:
        preview_path = args.initial_preview_clip
        reference_image = load_first_frame_image(preview_path)
        return reference_image, preview_path, "user_clip"

    global_clip_path = manifest_row.get("global_clip_path")
    if (
        isinstance(global_clip_path, str)
        and global_clip_path
        and os.path.exists(global_clip_path)
    ):
        reference_image = load_first_frame_image(global_clip_path)
        return reference_image, global_clip_path, "manifest_global_clip_first_frame"

    raise ValueError(
        "Could not resolve initial step reference image. "
        "Need `initial_frame_path` or `gold_frame_path` in manifest row, "
        "or pass --image_path/--initial_preview_clip."
    )


def build_user_initial_inputs(
    args, manifest_row: dict[str, Any] | None = None
) -> tuple[Image.Image, str, str]:
    if args.image_path:
        reference_image = Image.open(args.image_path).convert("RGB")
        return reference_image, args.image_path, "user_image"
    if args.initial_preview_clip:
        preview_path = args.initial_preview_clip
        reference_image = load_first_frame_image(preview_path)
        return reference_image, preview_path, "user_clip"
    if manifest_row is not None and args.origami_memory_mode == "mode2":
        global_clip_path = manifest_row.get("global_clip_path")
        if (
            isinstance(global_clip_path, str)
            and global_clip_path
            and os.path.exists(global_clip_path)
        ):
            reference_image = load_last_frame_image(global_clip_path)
            return (
                reference_image,
                global_clip_path,
                "manifest_global_clip_last_frame_training_aligned",
            )
    if manifest_row is not None:
        return build_initial_inputs_from_manifest_row(args, manifest_row)
    raise ValueError(
        "Could not resolve generated_prev step-1 image. Pass --image_path/--initial_preview_clip "
        "or provide a manifest row with gold_frame_path."
    )


def build_gold_frame_input_from_manifest_row(
    manifest_row: dict[str, Any]
) -> tuple[Image.Image, str, str]:
    gold_frame_path = manifest_row.get("gold_frame_path")
    if not (isinstance(gold_frame_path, str) and gold_frame_path):
        raise ValueError(
            "gold_frame_path is missing for manifest row: "
            f"{manifest_row.get('sample_id', '<unknown_sample_id>')}"
        )
    if not os.path.exists(gold_frame_path):
        raise FileNotFoundError(
            f"Missing gold_frame_path for row {manifest_row.get('sample_id', '<unknown_sample_id>')}: "
            f"{gold_frame_path}"
        )
    reference_image = Image.open(gold_frame_path).convert("RGB")
    return reference_image, gold_frame_path, "manifest_gold_frame_every_step"


def build_reference_clip_first_frame_input_from_manifest_row(
    manifest_row: dict[str, Any]
) -> tuple[Image.Image, str, str]:
    reference_clip_path = manifest_row.get("reference_clip_path")
    if not (isinstance(reference_clip_path, str) and reference_clip_path):
        raise ValueError(
            "reference_clip_path is missing for manifest row: "
            f"{manifest_row.get('sample_id', '<unknown_sample_id>')}"
        )
    if not os.path.exists(reference_clip_path):
        raise FileNotFoundError(
            f"Missing reference_clip_path for row {manifest_row.get('sample_id', '<unknown_sample_id>')}: "
            f"{reference_clip_path}"
        )
    reference_image = load_first_frame_image(reference_clip_path)
    return (
        reference_image,
        reference_clip_path,
        "manifest_reference_clip_first_frame_every_step",
    )


def build_later_step_inputs_with_mode(
    generated_clip_paths: list[str], reference_frame_mode: str
) -> tuple[Image.Image, str, str]:
    reference_clip_path = generated_clip_paths[-1]
    reference_image = load_reference_frame_image(
        reference_clip_path, frame_mode=reference_frame_mode
    )
    return reference_image, reference_clip_path, "generated_clip"


def build_continuous_bootstrap_input(
    args, manifest_row: dict[str, Any]
) -> tuple[Image.Image, str, str]:
    if args.reference_source == "generated_prev":
        return build_user_initial_inputs(args, manifest_row)
    if args.reference_source == "manifest_gold_every_step":
        return build_gold_frame_input_from_manifest_row(manifest_row)
    if args.reference_source == "manifest_reference_clip_first_every_step":
        return build_reference_clip_first_frame_input_from_manifest_row(manifest_row)
    try:
        return build_reference_clip_first_frame_input_from_manifest_row(manifest_row)
    except Exception:
        try:
            return build_gold_frame_input_from_manifest_row(manifest_row)
        except Exception:
            return build_initial_inputs_from_manifest_row(args, manifest_row)


def build_continuous_prompt(step_plan: list[dict[str, Any]]) -> str:
    step_lines = [step["step_prompt"] for step in step_plan]
    return "Generate one continuous tutorial video following these steps:\n" + "\n".join(
        step_lines
    )


def frame_windows_for_latent_blocks(
    step_count: int, step_latent_length: int, step_video_length: int
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    cursor = 0
    for step_idx in range(step_count):
        frame_count = step_video_length if step_idx == 0 else step_latent_length * 4
        windows.append((cursor, cursor + frame_count))
        cursor += frame_count
    return windows


def generate_continuous_video(
    pipe: HunyuanVideo_1_5_Pipeline,
    output_root: Path,
    args,
) -> dict[str, Any]:
    if args.reference_source in (
        "manifest_gold_every_step",
        "manifest_reference_clip_first_every_step",
    ):
        raise ValueError(
            "continuity_mode=latent_memory uses one continuous pipeline call and cannot "
            f"apply reference_source={args.reference_source!r} at every step. "
            "Use --reference_source generated_prev with --image_path/--initial_preview_clip "
            "for generated-state continuity, or --reference_source train_aligned_auto if you "
            "only want the first manifest visual as the initial bootstrap image."
        )

    clips_dir = output_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_root / args.final_output_name

    step_plan, stop_reason = load_manifest_step_plan(args)
    if not step_plan:
        raise RuntimeError("No steps were found in the manifest step plan.")

    step_latent_length = latent_length_from_video_length(args.step_video_length)
    requested_latent_length = step_latent_length * len(step_plan)
    chunk_latent_frames = args.chunk_latent_frames or 4
    if args.model_type == "ar":
        if chunk_latent_frames != 4:
            rank0_log(
                "continuity_mode=latent_memory follows generate.py AR rollout and "
                f"uses chunk_latent_frames=4 instead of requested {chunk_latent_frames}."
            )
        chunk_latent_frames = 4
    if chunk_latent_frames <= 0:
        raise ValueError("chunk_latent_frames must be positive.")
    padded_latent_length = (
        (requested_latent_length + chunk_latent_frames - 1)
        // chunk_latent_frames
        * chunk_latent_frames
    )
    total_video_length = (padded_latent_length - 1) * 4 + 1

    first_row = step_plan[0]["manifest_row"]
    reference_image, reference_visual_path, initial_visual_source_type = (
        build_continuous_bootstrap_input(args, first_row)
    )
    continuous_prompt = build_continuous_prompt(step_plan)
    viewmats, intrinsics, action = build_static_controls(padded_latent_length)

    generation_output = None
    decoded = None
    final_written_frames: int | None = None
    step_results: list[dict[str, Any]] = []
    try:
        generation_output = pipe(
            prompt=continuous_prompt,
            aspect_ratio=args.aspect_ratio,
            video_length=total_video_length,
            prompt_rewrite=False,
            num_inference_steps=args.num_inference_steps,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            flow_shift=args.flow_shift,
            embedded_guidance_scale=args.embedded_guidance_scale,
            reference_image=reference_image,
            user_height=args.height,
            user_width=args.width,
            chunk_latent_frames=chunk_latent_frames,
            guidance_scale=args.guidance_scale,
            viewmats=viewmats,
            Ks=intrinsics,
            action=action,
            enable_sr=False,
            output_type="pt",
            return_dict=True,
            few_step=args.few_step,
            model_type=args.model_type,
            transformer_resident_ar_rollout=args.transformer_resident_ar_rollout,
            memory_frames=args.memory_frames,
            temporal_context_size=args.temporal_context_size,
            memory_frame_policy=args.memory_frame_policy,
            origami_memory_mode=args.origami_memory_mode,
            origami_memory_steps=args.origami_memory_steps,
            origami_memory_policy=args.origami_memory_policy,
            origami_memory_blend=args.origami_memory_blend,
        )
        decoded = generation_output.videos
        if decoded is None:
            raise RuntimeError("Pipeline returned empty video for continuous prompt.")

        frame_windows = frame_windows_for_latent_blocks(
            len(step_plan), step_latent_length, args.step_video_length
        )
        used_frame_count = frame_windows[-1][1]

        if is_rank0():
            save_video(decoded[:, :, :used_frame_count], final_output_path, args.output_fps)
            final_written_frames = used_frame_count

            for step, (start_frame, end_frame) in zip(step_plan, frame_windows):
                step_number = step["step_number"]
                clip_output_path = clips_dir / f"step_{step_number:02d}.mp4"
                save_video(
                    decoded[:, :, start_frame:end_frame],
                    clip_output_path,
                    args.output_fps,
                )
                step_results.append(
                    {
                        "step_number": step_number,
                        "raw_step_prompt": step["raw_step_prompt"],
                        "step_prompt": step["step_prompt"],
                        "step_text": step["step_text"],
                        "clip_path": str(clip_output_path),
                        "reference_visual_path": reference_visual_path,
                        "initial_visual_source_type": initial_visual_source_type,
                        "user_image_path": args.image_path,
                        "user_initial_preview_clip": args.initial_preview_clip,
                        "manifest_initial_frame_path": step["manifest_row"].get(
                            "initial_frame_path"
                        ),
                        "manifest_global_clip_path": step["manifest_row"].get(
                            "global_clip_path"
                        ),
                        "manifest_gold_frame_path": step["manifest_row"].get(
                            "gold_frame_path"
                        ),
                        "manifest_reference_clip_path": step["manifest_row"].get(
                            "reference_clip_path"
                        ),
                        "generated_frames_this_step": end_frame - start_frame,
                        "cumulative_generated_frames_after_step": end_frame,
                        "step_seed": args.seed,
                        "continuity_mode": args.continuity_mode,
                        "stitch_drop_first_after_step1": False,
                        "video_length": end_frame - start_frame,
                        "latent_length": step_latent_length,
                        "output_path": str(clip_output_path),
                        "frame_start": start_frame,
                        "frame_end": end_frame,
                    }
                )
            rank0_log(
                f"Saved continuous video to {final_output_path} | "
                f"written_frames={used_frame_count}"
            )
        stage_barrier()

        summary = {
            "manifest_path": args.manifest_path,
            "prompt_manifest_path": step_plan[0]["prompt_manifest_path"],
            "video_id": args.video_id,
            "reference_source": args.reference_source,
            "action_ckpt": args.action_ckpt,
            "strict_checkpoint_load": True,
            "user_image_path": args.image_path,
            "user_initial_preview_clip": args.initial_preview_clip,
            "step_prompt_count_in_plan": len(step_plan),
            "generated_step_count": len(step_plan),
            "stop_reason": stop_reason,
            "step_video_length": args.step_video_length,
            "total_generated_frames": used_frame_count,
            "final_written_frames": final_written_frames,
            "output_fps": args.output_fps,
            "final_output_path": str(final_output_path),
            "continuity": {
                "mode": args.continuity_mode,
                "single_pipe_call": True,
                "seed_schedule": "single_call_fixed",
                "step_seeds": [args.seed for _ in step_plan],
                "prompt_memory_steps": 0,
                "memory_frames": args.memory_frames,
                "temporal_context_size": args.temporal_context_size,
                "memory_frame_policy": args.memory_frame_policy,
                "reference_frame_mode": "not_used_single_call",
                "reference_policy": "initial_bootstrap_only",
                "strict_generated_prev_step1_user_source": args.reference_source
                == "generated_prev",
                "stitch_drop_first_after_step1": False,
                "requested_latent_length": requested_latent_length,
                "padded_latent_length": padded_latent_length,
                "chunk_latent_frames": chunk_latent_frames,
                "continuous_prompt": continuous_prompt,
            },
            "steps": step_results,
        }
        if is_rank0() and args.save_summary_json:
            summary_path = output_root / "step_prompt_summary.json"
            with open(summary_path, "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)
            rank0_log(f"Saved step-prompt summary to {summary_path}")
        return summary
    finally:
        del generation_output
        del decoded
        cleanup_tensors()


def run_step_prompt_generation(
    pipe: HunyuanVideo_1_5_Pipeline,
    output_root: Path,
    args,
) -> dict[str, Any]:
    clips_dir = output_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_root / args.final_output_name

    step_plan, stop_reason = load_manifest_step_plan(args)

    generated_clip_paths: list[str] = []
    step_results: list[dict[str, Any]] = []
    cumulative_generated_frames = 0
    final_written_frames: int | None = None
    continuity_mode = args.continuity_mode
    prompt_history: list[str] = []
    step_seeds: list[int] = []
    generated_step_latents: list[torch.Tensor] = []
    generated_reference_images: list[Image.Image] = []

    for step_idx, step in enumerate(step_plan):
        step_number = step["step_number"]
        manifest_row = step["manifest_row"]
        raw_step = step["raw_step_prompt"]
        is_first_step = step_idx == 0
        use_strict_generated_prev_bootstrap = (
            args.reference_source == "generated_prev" and is_first_step
        )

        if args.reference_source == "manifest_gold_every_step":
            reference_image, reference_visual_path, initial_visual_source_type = (
                build_gold_frame_input_from_manifest_row(manifest_row)
            )
        elif args.reference_source == "manifest_reference_clip_first_every_step":
            reference_image, reference_visual_path, initial_visual_source_type = (
                build_reference_clip_first_frame_input_from_manifest_row(manifest_row)
            )
        elif args.reference_source == "train_aligned_auto":
            try:
                (
                    reference_image,
                    reference_visual_path,
                    initial_visual_source_type,
                ) = build_reference_clip_first_frame_input_from_manifest_row(manifest_row)
            except Exception:
                try:
                    (
                        reference_image,
                        reference_visual_path,
                        initial_visual_source_type,
                    ) = build_gold_frame_input_from_manifest_row(manifest_row)
                except Exception:
                    if generated_clip_paths:
                        (
                            reference_image,
                            reference_visual_path,
                            initial_visual_source_type,
                        ) = build_later_step_inputs_with_mode(
                            generated_clip_paths, args.reference_frame_mode
                        )
                    else:
                        (
                            reference_image,
                            reference_visual_path,
                            initial_visual_source_type,
                        ) = build_initial_inputs_from_manifest_row(args, manifest_row)
        elif generated_clip_paths:
            reference_image, reference_visual_path, initial_visual_source_type = (
                build_later_step_inputs_with_mode(
                    generated_clip_paths, args.reference_frame_mode
                )
            )
        elif use_strict_generated_prev_bootstrap:
            reference_image, reference_visual_path, initial_visual_source_type = (
                build_user_initial_inputs(args, manifest_row)
            )
        else:
            reference_image, reference_visual_path, initial_visual_source_type = (
                build_initial_inputs_from_manifest_row(args, manifest_row)
            )

        step_prompt = step["step_prompt"]
        step_text = step["step_text"]
        if continuity_mode == "stateless_prompt_ref" and args.prompt_memory_steps > 0:
            memory_steps = prompt_history[-args.prompt_memory_steps :]
            if memory_steps:
                memory_prefix = "Previous steps context:\n" + "\n".join(
                    f"- {txt}" for txt in memory_steps
                )
                step_prompt = f"{memory_prefix}\n\nCurrent step:\n{step_prompt}"

        if args.seed_schedule == "increment":
            step_seed = args.seed + step_idx
        else:
            step_seed = args.seed
        step_seeds.append(step_seed)

        memory_history_latents = None
        if is_first_step:
            memory_history_latents = build_global_clip_memory_latents(
                pipe=pipe,
                global_clip_path=str(manifest_row.get("global_clip_path") or ""),
                latent_length=latent_length_from_video_length(args.step_video_length),
                args=args,
            )
            if memory_history_latents is not None and os.getenv("ORIGAMI_DEBUG", "0") == "1":
                rank0_log(
                    f"[memory-debug] step={step_number} image_cond_source=global_clip_path "
                    f"memory_history_shape={tuple(memory_history_latents.shape)}"
                )
            elif os.getenv("ORIGAMI_DEBUG", "0") == "1":
                rank0_log(
                    f"[memory-debug] step={step_number} image_cond_source=frame0_image"
                )
        elif generated_step_latents and args.origami_memory_mode == "mode2":
            if args.origami_memory_policy != "latest_k":
                raise ValueError(
                    f"Unsupported origami_memory_policy={args.origami_memory_policy!r}; expected 'latest_k'."
                )
            memory_steps = max(int(args.origami_memory_steps), 1)
            selected_latents = generated_step_latents[-memory_steps:]
            if len(selected_latents) < memory_steps and is_rank0():
                print(
                    f"[origami-memory] WARNING step={step_number}: requested memory_steps={memory_steps} "
                    f"but only {len(selected_latents)} clips available, using {len(selected_latents)}",
                    flush=True,
                )
            target_hist_frames = max(
                latent_length_from_video_length(args.step_video_length) - 1, 0
            )
            memory_history_latents = pad_or_trim_memory_latents(
                torch.cat(selected_latents, dim=2), target_hist_frames
            ) * args.origami_memory_blend
            memory_nonzero = memory_history_latents.abs().sum().item()
            if os.getenv("ORIGAMI_DEBUG", "0") == "1":
                rank0_log(
                    f"[memory-debug] step={step_number} image_cond_source=past_latents "
                    f"memory_history_shape={tuple(memory_history_latents.shape)} "
                    f"memory_frames={memory_history_latents.shape[2]} "
                    f"memory_nonzero={memory_nonzero:.6f} "
                    f"blend={args.origami_memory_blend}"
                )

        clip_output_path = clips_dir / f"step_{step_number:02d}.mp4"
        generation_result = generate_step_clip(
            pipe=pipe,
            step_prompt=step_prompt,
            reference_image=reference_image,
            video_length=args.step_video_length,
            output_path=clip_output_path,
            step_seed=step_seed,
            args=args,
            memory_history_latents=memory_history_latents,
            memory_reference_images=generated_reference_images,
        )
        generated_latents = generation_result.pop("_generated_latents")
        generated_step_latents.append(generated_latents)
        generated_reference_images.append(reference_image.copy())
        generated_clip_paths.append(str(clip_output_path))
        prompt_history.append(step_text)
        cumulative_generated_frames += args.step_video_length

        step_results.append(
            {
                "step_number": step_number,
                "raw_step_prompt": raw_step,
                "step_prompt": step_prompt,
                "step_text": step_text,
                "clip_path": str(clip_output_path),
                "reference_visual_path": reference_visual_path,
                "initial_visual_source_type": initial_visual_source_type,
                "user_image_path": args.image_path,
                "user_initial_preview_clip": args.initial_preview_clip,
                "manifest_initial_frame_path": manifest_row.get("initial_frame_path"),
                "manifest_global_clip_path": manifest_row.get("global_clip_path"),
                "manifest_gold_frame_path": manifest_row.get("gold_frame_path"),
                "manifest_reference_clip_path": manifest_row.get("reference_clip_path"),
                "generated_frames_this_step": args.step_video_length,
                "cumulative_generated_frames_after_step": cumulative_generated_frames,
                "step_seed": step_seed,
                "continuity_mode": continuity_mode,
                "stitch_drop_first_after_step1": args.stitch_drop_first_after_step1,
                **generation_result,
            }
        )
        rank0_log(
            f"Generated step {step_number}/{len(step_plan)} | "
            f"step_prompt={step_prompt} | "
            f"seed={step_seed} | "
            f"reference_source={initial_visual_source_type} | "
            f"reference_visual_path={reference_visual_path} | "
            f"cumulative_generated_frames={cumulative_generated_frames}"
        )

    if not generated_clip_paths:
        raise RuntimeError("No clips were generated from the manifest step plan.")

    if is_rank0():
        writer = imageio.get_writer(final_output_path, fps=args.output_fps)
        written_frames = 0
        try:
            for clip_idx, clip_path in enumerate(generated_clip_paths):
                reader = imageio.get_reader(clip_path)
                try:
                    for frame_idx, frame in enumerate(reader):
                        if (
                            args.stitch_drop_first_after_step1
                            and clip_idx > 0
                            and frame_idx == 0
                        ):
                            continue
                        writer.append_data(frame)
                        written_frames += 1
                finally:
                    reader.close()
        finally:
            writer.close()
        rank0_log(
            f"Saved final video to {final_output_path} | "
            f"written_frames={written_frames}"
        )
        final_written_frames = written_frames

    summary = {
        "manifest_path": args.manifest_path,
        "prompt_manifest_path": step_plan[0]["prompt_manifest_path"] if step_plan else "",
        "video_id": args.video_id,
        "reference_source": args.reference_source,
        "action_ckpt": args.action_ckpt,
        "strict_checkpoint_load": True,
        "user_image_path": args.image_path,
        "user_initial_preview_clip": args.initial_preview_clip,
        "step_prompt_count_in_plan": len(step_plan),
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
            "strict_generated_prev_step1_user_source": args.reference_source == "generated_prev",
            "stitch_drop_first_after_step1": args.stitch_drop_first_after_step1,
        },
        "steps": step_results,
    }
    if is_rank0() and args.save_summary_json:
        summary_path = output_root / "step_prompt_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        rank0_log(f"Saved step-prompt summary to {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Free-run origami generation from manifest rows for a specific video_id."
    )
    parser.add_argument("--action_ckpt", type=str, default=_default_action_ckpt())
    parser.add_argument("--model_path", type=str, default=_default_model_path())
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--initial_preview_clip", type=str, default=None)
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Manifest path containing rows for many videos (must include video_id, step_number; optionally gold_frame_path).",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        required=True,
        help="Video ID to generate step-by-step (for example: paper_boat_15).",
    )
    parser.add_argument(
        "--prompt_manifest_path",
        type=str,
        default=None,
        help=(
            "Optional manifest used for prompt lookup if manifest_path lacks step captions. "
            "Must include per-step text such as target_caption."
        ),
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--final_output_name",
        type=str,
        default="gen.mp4",
    )
    parser.add_argument("--save_summary_json", type=str_to_bool, nargs="?", const=True, default=True)

    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    parser.add_argument("--aspect_ratio", type=str, default="16:9")
    parser.add_argument(
        "--video_length",
        type=int,
        default=None,
        help="Optional maximum number of steps/clips to generate. Defaults to all steps in the file.",
    )
    parser.add_argument(
        "--step_video_length",
        type=int,
        default=77,
        help="Number of frames to generate for each step clip. Use 77 to match origami training defaults.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3208)
    parser.add_argument(
        "--seed_schedule",
        type=str,
        default="increment",
        choices=list(SEED_SCHEDULES),
        help="How to set per-step seeds: fixed uses same seed, increment uses seed + step_idx.",
    )
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=None,
        help="Override scheduler flow shift. None uses pipeline default for the selected model version.",
    )
    parser.add_argument(
        "--embedded_guidance_scale",
        type=float,
        default=None,
        help="Optional embedded guidance scale; keep None unless your model supports guidance embedding.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--output_fps", type=int, default=16)
    parser.add_argument(
        "--chunk_latent_frames",
        type=int,
        default=4,
        help=(
            "Latent chunk size for AR rollout. "
            "Use 4 to match training behavior. "
            "Set to step latent length to force single-chunk inference."
        ),
    )
    parser.add_argument(
        "--reference_frame_mode",
        type=str,
        default="last",
        choices=["first", "last"],
        help=(
            "Which frame from the previous generated clip to condition the next step on. "
            "'last' is best for chained continuity."
        ),
    )
    parser.add_argument(
        "--reference_source",
        type=str,
        default="generated_prev",
        choices=[
            "generated_prev",
            "manifest_gold_every_step",
            "manifest_reference_clip_first_every_step",
            "train_aligned_auto",
        ],
        help=(
            "Reference-image source for each step. "
            "`generated_prev` chains from previously generated clip; "
            "`manifest_gold_every_step` uses each row's gold_frame_path for every step; "
            "`manifest_reference_clip_first_every_step` uses the first frame from each row's reference_clip_path "
            "(matches origami precompute training reference policy); "
            "`train_aligned_auto` prefers manifest reference_clip_path, falls back to gold frame, then generated_prev."
        ),
    )
    parser.add_argument(
        "--continuity_mode",
        type=str,
        default="stateless_prompt_ref",
        choices=list(CONTINUITY_MODES),
        help=(
            "Continuity strategy. stateless_prompt_ref works on current pipeline; "
            "latent_memory uses one continuous pipeline call, matching generate.py AR rollout, "
            "then slices the result into per-step clips."
        ),
    )
    parser.add_argument(
        "--prompt_memory_steps",
        type=int,
        default=0,
        help="Number of prior step texts to prepend as prompt context in stateless continuity mode.",
    )
    parser.add_argument(
        "--memory_frames",
        type=int,
        default=20,
        help="Requested memory frames for latent continuation mode (for compatible patched pipelines).",
    )
    parser.add_argument(
        "--temporal_context_size",
        type=int,
        default=12,
        help="Requested temporal context window for latent continuation mode (for compatible patched pipelines).",
    )
    parser.add_argument(
        "--memory_frame_policy",
        type=str,
        default="recent",
        choices=["recent", "aligned"],
        help=(
            "How AR rollout selects history frames. recent uses only the last --memory_frames "
            "latent frames to reduce VRAM; aligned uses the original retrieval-based selector."
        ),
    )
    parser.add_argument("--origami_memory_mode", type=str, default=os.getenv("ORIGAMI_MEMORY_MODE", "mode2"))
    parser.add_argument("--origami_memory_steps", type=int, default=int(os.getenv("ORIGAMI_MEMORY_STEPS", "2")))
    parser.add_argument("--origami_memory_policy", type=str, default=os.getenv("ORIGAMI_MEMORY_POLICY", "latest_k"))
    parser.add_argument("--origami_memory_blend", type=float, default=float(os.getenv("ORIGAMI_MEMORY_BLEND", "1.0")))
    parser.add_argument(
        "--save_step_latents",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Save step-level latent state in latent_memory mode when supported.",
    )
    parser.add_argument(
        "--stitch_drop_first_after_step1",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Drop first frame of clips after step 1 while stitching to reduce boundary seams.",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument(
        "--offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--group_offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ar",
        choices=["ar", "bi"],
        help=(
            "Rollout used by the pipeline. 'ar' is the KV-cached autoregressive path "
            "(forward_vision), which ignores compressed memory tokens. 'bi' runs the "
            "full-window path (forward_bi), which consumes them and therefore matches "
            "the training-time forward. Use 'bi' with origami_memory_mode=mode2."
        ),
    )
    parser.add_argument(
        "--enable_torch_compile",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--use_sageattn",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--sage_blocks_range", type=str, default="0-53")
    parser.add_argument(
        "--use_vae_parallel",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--use_fp8_gemm",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--quant_type", type=str, default="fp8-per-block")
    parser.add_argument("--include_patterns", type=str, default="double_blocks")
    parser.add_argument(
        "--few_step",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--transformer_resident_ar_rollout",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Keep transformer resident during AR rollout, matching generate.py. "
            "Only affects AR model_type with offloading enabled."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(normalize_leading_dash_arg_values(sys.argv[1:]))

    validate_initial_inputs(args)
    if args.step_video_length <= 0:
        raise ValueError("step_video_length must be positive.")
    if args.chunk_latent_frames is not None and args.chunk_latent_frames <= 0:
        raise ValueError("chunk_latent_frames must be positive.")
    if args.video_length is not None and args.video_length <= 0:
        raise ValueError("video_length must be positive when provided.")
    if args.prompt_memory_steps < 0:
        raise ValueError("prompt_memory_steps must be >= 0.")
    if args.memory_frames <= 0:
        raise ValueError("memory_frames must be positive.")
    if args.temporal_context_size <= 0:
        raise ValueError("temporal_context_size must be positive.")
    if args.continuity_mode == "latent_memory" and args.save_step_latents:
        raise NotImplementedError(
            "--save_step_latents is not available in continuity_mode=latent_memory yet; "
            "the current implementation saves decoded per-step clips from one continuous video."
        )

    initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    initialize_infer_state(args)

    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    rank0_log(
        f"Starting fixed-step origami generation | action_ckpt={args.action_ckpt} | "
        f"manifest_path={args.manifest_path} | video_id={args.video_id} | "
        f"step_video_length={args.step_video_length}"
    )
    rank0_log(
        f"[origami-memory] mode={args.origami_memory_mode}, steps={args.origami_memory_steps}, "
        f"policy={args.origami_memory_policy}, blend={args.origami_memory_blend}, "
        f"model_type={args.model_type}"
    )
    if args.origami_memory_mode == "mode2" and args.model_type != "bi":
        rank0_log(
            "[origami-memory] WARNING: origami_memory_mode=mode2 with model_type="
            f"{args.model_type}. The autoregressive path computes the compressed "
            "memory tokens and then discards them, so this run is NOT equivalent to "
            "training. Use --model_type bi to consume them."
        )
    if args.origami_memory_mode != "mode2" and args.origami_memory_blend != 1.0:
        rank0_log(
            "[origami-memory] NOTE: origami_memory_blend only affects mode2; "
            f"value {args.origami_memory_blend} is ignored in "
            f"{args.origami_memory_mode}."
        )

    pipe = None
    try:
        pipe = create_pipeline(args)
        if args.continuity_mode == "latent_memory":
            generate_continuous_video(
                pipe=pipe,
                output_root=output_root,
                args=args,
            )
        else:
            run_step_prompt_generation(
                pipe=pipe,
                output_root=output_root,
                args=args,
            )
        stage_barrier()
    finally:
        del pipe
        cleanup_tensors(finalize_distributed=True)


if __name__ == "__main__":
    main()
