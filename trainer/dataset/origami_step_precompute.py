# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
from PIL import Image
from tqdm import tqdm

# Allow execution via an absolute script path without requiring callers to
# pre-populate PYTHONPATH with the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyvideo.models.autoencoders import hunyuanvideo_15_vae_w_cache
from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from trainer.dataset.origami_precompute_utils import (
    SUPPORTED_SHARD_STRATEGIES,
    atomic_json_dump,
    build_normalized_row,
    build_shard_assignments,
    choose_reference_clip,
    format_origami_text_prompt,
    flush_rank_outputs,
    get_progress_dir,
    get_rank_progress_path,
    get_shard_dir,
    merge_rank_outputs,
    summarize_assignment_costs,
)
from trainer.dataset.transform import CenterCropResizeVideo


@dataclass
class OrigamiPrecomputeConfig:
    raw_manifest: str
    output_manifest: str
    feature_cache_dir: str
    model_path: str
    device: str = "auto"
    height: int = 480
    width: int = 832
    target_fps: int = 24
    history_keep: int = 3
    reference_mode: str = "latest_past_or_global"
    overwrite: bool = False
    save_every: int = 100
    progress_every: int = 10
    shard_strategy: str = "balanced_duration"
    dist_timeout_seconds: int = 7200


@dataclass
class DistributedRuntime:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    enabled: bool = False
    initialized_here: bool = False


class OrigamiFeatureEncoder:
    def __init__(self, config: OrigamiPrecomputeConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.video_transform = CenterCropResizeVideo((config.height, config.width))
        self.vae = self._load_vae(config.model_path, self.device)
        self.text_encoder, _ = HunyuanVideo_1_5_Pipeline._load_text_encoders(
            config.model_path, device=self.device
        )
        self.byt5_kwargs, self.prompt_format = HunyuanVideo_1_5_Pipeline._load_byt5(
            config.model_path, True, 256, device=self.device
        )
        self.vision_encoder = HunyuanVideo_1_5_Pipeline._load_vision_encoder(
            config.model_path, device=self.device
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cpu")
        if device == "cuda":
            return torch.device(f"cuda:{local_rank}")
        return torch.device(device)

    @staticmethod
    def _load_vae(model_path: str, device: torch.device):
        return hunyuanvideo_15_vae_w_cache.AutoencoderKLConv3D.from_pretrained(
            os.path.join(model_path, "vae"), torch_dtype=torch.float16
        ).to(device)

    @staticmethod
    def _sample_frame_indices(total_frames: int, source_fps: float, target_fps: int):
        if total_frames <= 0:
            return np.array([], dtype=np.int64)
        if not source_fps or source_fps <= 0:
            return np.arange(total_frames, dtype=np.int64)

        frame_interval = source_fps / float(target_fps)
        if frame_interval <= 1.0:
            return np.arange(total_frames, dtype=np.int64)

        indices = np.arange(0, total_frames, frame_interval).astype(np.int64)
        return np.clip(indices, 0, total_frames - 1)

    def _load_video_tensor(self, video_path: str) -> tuple[torch.Tensor, float]:
        frames, _, metadata = torchvision.io.read_video(video_path, output_format="TCHW")
        source_fps = float(metadata.get("video_fps", 0.0) or 0.0)
        sample_indices = self._sample_frame_indices(
            total_frames=frames.shape[0],
            source_fps=source_fps,
            target_fps=self.config.target_fps,
        )
        sampled = frames[sample_indices]
        sampled = self.video_transform(sampled)
        sampled = rearrange(sampled, "t c h w -> c t h w").float() / 127.5 - 1.0
        return sampled, source_fps

    def _load_last_frame_image(self, clip_path: str) -> Image.Image:
        frames, _, _ = torchvision.io.read_video(clip_path, output_format="TCHW")
        if frames.shape[0] == 0:
            raise ValueError(f"No frames found in reference clip: {clip_path}")
        last_frame = frames[-1].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(last_frame)

    def _encode_video_latent(self, clip_path: str) -> torch.Tensor:
        video_tensor, _ = self._load_video_tensor(clip_path)
        video_tensor = video_tensor.unsqueeze(0).to(self.device, dtype=self.vae.dtype)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            latent = self.vae.encode(video_tensor).latent_dist.mode()
            latent = latent * self.vae.config.scaling_factor
        return latent.cpu()

    def _encode_reference_latent(self, reference_image: Image.Image) -> torch.Tensor:
        image_np = np.array(reference_image.convert("RGB"))
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        image_tensor = self.video_transform(image_tensor)
        image_tensor = image_tensor.transpose(0, 1).float() / 127.5 - 1.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device, dtype=self.vae.dtype)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            image_cond = self.vae.encode(image_tensor).latent_dist.mode()
            image_cond = image_cond * self.vae.config.scaling_factor
        return image_cond.cpu()

    def _encode_text(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.text_encoder.text2tokens(
            prompt, data_type="video", max_length=self.text_encoder.max_length
        )
        with torch.inference_mode():
            outputs = self.text_encoder.encode(tokens, data_type="video", device=self.device)
        return outputs.hidden_state.cpu(), outputs.attention_mask.cpu()

    def _encode_byt5(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        byt5_embeddings = torch.zeros(
            (1, self.byt5_kwargs["byt5_max_length"], 1472), dtype=self.vae.dtype
        )
        byt5_mask = torch.zeros(
            (1, self.byt5_kwargs["byt5_max_length"]), dtype=torch.int64
        )

        if "\"" not in prompt and "“" not in prompt:
            return byt5_embeddings.cpu(), byt5_mask.cpu()

        glyph_texts = []
        for quote in ('"', "“"):
            if quote in prompt:
                parts = prompt.split(quote)
                glyph_texts.extend(parts[1::2])
        glyph_texts = list(dict.fromkeys([text for text in glyph_texts if text]))
        if not glyph_texts:
            return byt5_embeddings.cpu(), byt5_mask.cpu()

        text_styles = [{"color": None, "font-family": None} for _ in glyph_texts]
        formatted = self.prompt_format.format_prompt(glyph_texts, text_styles)
        text_ids, text_mask = HunyuanVideo_1_5_Pipeline.get_byt5_text_tokens(
            self.byt5_kwargs["byt5_tokenizer"],
            self.byt5_kwargs["byt5_max_length"],
            formatted,
        )
        text_ids = text_ids.to(self.device)
        text_mask = text_mask.to(self.device)
        with torch.inference_mode():
            outputs = self.byt5_kwargs["byt5_model"](
                text_ids, attention_mask=text_mask.float()
            )
        byt5_embeddings = outputs[0].cpu()
        byt5_mask = text_mask.cpu()
        return byt5_embeddings, byt5_mask

    def _encode_vision_states(self, reference_image: Image.Image) -> torch.Tensor:
        with torch.inference_mode():
            image_np = np.array(reference_image.convert("RGB"))
            outputs = self.vision_encoder.encode_images(image_np)
        return outputs.last_hidden_state.cpu()

    def build_negative_text_features(self) -> dict[str, torch.Tensor]:
        prompt_embeds, prompt_mask = self._encode_text("")
        byt5_text_states, byt5_text_mask = self._encode_byt5("")
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_mask": prompt_mask,
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
        }

    def choose_reference_clip(self, sample: dict) -> str:
        return choose_reference_clip(sample, self.config.reference_mode)

    def encode_sample(self, sample: dict) -> tuple[dict[str, torch.Tensor], str]:
        prompt = format_origami_text_prompt(sample)
        reference_clip_path = self.choose_reference_clip(sample)
        reference_image = self._load_last_frame_image(reference_clip_path)

        latent = self._encode_video_latent(sample["clip_path"])
        prompt_embeds, prompt_mask = self._encode_text(prompt)
        byt5_text_states, byt5_text_mask = self._encode_byt5(prompt)
        image_cond = self._encode_reference_latent(reference_image)
        vision_states = self._encode_vision_states(reference_image)

        feature_pt = {
            "latent": latent,
            "prompt_embeds": prompt_embeds,
            "prompt_mask": prompt_mask,
            "image_cond": image_cond,
            "vision_states": vision_states,
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
        }
        return feature_pt, reference_clip_path


def _load_manifest(manifest_path: str) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _atomic_torch_save(payload, output_path: str) -> None:
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=output_dir, suffix=".tmp"
    ) as fp:
        temp_path = fp.name
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _maybe_init_distributed(
    device: str, dist_timeout_seconds: int
) -> DistributedRuntime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    runtime = DistributedRuntime(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        enabled=world_size > 1,
        initialized_here=False,
    )
    if not runtime.enabled:
        return runtime

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable but WORLD_SIZE > 1.")

    if not dist.is_initialized():
        resolved_device = OrigamiFeatureEncoder._resolve_device(device)
        backend = "gloo"
        if resolved_device.type == "cuda":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=dist_timeout_seconds),
        )
        runtime.initialized_here = True

    if OrigamiFeatureEncoder._resolve_device(device).type == "cuda":
        torch.cuda.set_device(runtime.local_rank)
    return runtime


def _destroy_distributed(runtime: DistributedRuntime) -> None:
    if runtime.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def _barrier(runtime: DistributedRuntime) -> None:
    if runtime.enabled and dist.is_initialized():
        dist.barrier()


def _format_rank_sample_preview(
    manifest: list[dict], shard_indices: list[int], preview_count: int = 3
) -> list[str]:
    preview_indices = shard_indices[:preview_count]
    return [manifest[idx]["sample_id"] for idx in preview_indices]


def _write_rank_progress(
    output_manifest: str,
    rank: int,
    *,
    assigned_samples: int,
    completed_samples: int,
    missing_features_total: int,
    missing_features_completed: int,
) -> None:
    atomic_json_dump(
        {
            "rank": rank,
            "assigned_samples": assigned_samples,
            "completed_samples": completed_samples,
            "missing_features_total": missing_features_total,
            "missing_features_completed": missing_features_completed,
        },
        get_rank_progress_path(output_manifest, rank),
    )


def _read_progress_summary(output_manifest: str, world_size: int) -> dict[str, int]:
    summary = {
        "assigned_samples": 0,
        "completed_samples": 0,
        "missing_features_total": 0,
        "missing_features_completed": 0,
        "reporting_ranks": 0,
    }
    for rank in range(world_size):
        progress_path = get_rank_progress_path(output_manifest, rank)
        if not os.path.exists(progress_path):
            continue
        with open(progress_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        summary["assigned_samples"] += int(payload.get("assigned_samples", 0))
        summary["completed_samples"] += int(payload.get("completed_samples", 0))
        summary["missing_features_total"] += int(payload.get("missing_features_total", 0))
        summary["missing_features_completed"] += int(
            payload.get("missing_features_completed", 0)
        )
        summary["reporting_ranks"] += 1
    return summary


def _refresh_global_progress_bar(
    progress_bar: tqdm | None,
    output_manifest: str,
    world_size: int,
) -> None:
    if progress_bar is None:
        return

    summary = _read_progress_summary(output_manifest, world_size)
    completed = summary["missing_features_completed"]
    if completed > progress_bar.n:
        progress_bar.update(completed - progress_bar.n)

    remaining = max(summary["missing_features_total"] - completed, 0)
    progress_bar.set_postfix(
        remaining=remaining,
        processed=summary["completed_samples"],
        ranks=f"{summary['reporting_ranks']}/{world_size}",
    )


def build_origami_training_cache(config: OrigamiPrecomputeConfig) -> None:
    runtime = _maybe_init_distributed(config.device, config.dist_timeout_seconds)
    try:
        raw_manifest = _load_manifest(config.raw_manifest)
        os.makedirs(config.feature_cache_dir, exist_ok=True)
        output_manifest_dir = os.path.dirname(config.output_manifest)
        if output_manifest_dir:
            os.makedirs(output_manifest_dir, exist_ok=True)
        os.makedirs(get_shard_dir(config.output_manifest), exist_ok=True)
        os.makedirs(get_progress_dir(config.output_manifest), exist_ok=True)

        negative_feature_path = os.path.join(config.feature_cache_dir, "negative_prompt.pt")
        shard_assignments = build_shard_assignments(
            manifest=raw_manifest,
            world_size=runtime.world_size,
            shard_strategy=config.shard_strategy,
        )
        shard_indices = shard_assignments[runtime.rank]
        shard_samples = [
            (manifest_index, raw_manifest[manifest_index])
            for manifest_index in shard_indices
        ]
        rank_costs = summarize_assignment_costs(raw_manifest, shard_assignments)
        assigned_total_cost = rank_costs[runtime.rank]

        local_missing_feature_count = 0
        for _, sample in shard_samples:
            feature_pt_path = os.path.join(
                config.feature_cache_dir, f"{sample['sample_id']}.pt"
            )
            if config.overwrite or not os.path.exists(feature_pt_path):
                local_missing_feature_count += 1

        _write_rank_progress(
            config.output_manifest,
            runtime.rank,
            assigned_samples=len(shard_samples),
            completed_samples=0,
            missing_features_total=local_missing_feature_count,
            missing_features_completed=0,
        )
        _barrier(runtime)
        progress_summary = _read_progress_summary(
            config.output_manifest, runtime.world_size
        )
        global_missing_feature_count = progress_summary["missing_features_total"]

        needs_negative_features = config.overwrite or not os.path.exists(
            negative_feature_path
        )
        should_build_encoder = local_missing_feature_count > 0 or (
            runtime.rank == 0 and needs_negative_features
        )

        print(
            f"[rank {runtime.rank}/{runtime.world_size}] assigned {len(shard_samples)} samples, "
            f"shard_strategy={config.shard_strategy}, "
            f"estimated_cost={assigned_total_cost:.2f}, "
            f"missing_features={local_missing_feature_count}, "
            f"global_missing_features={global_missing_feature_count}, "
            f"needs_negative_features={bool(runtime.rank == 0 and needs_negative_features)}, "
            f"sample_preview={_format_rank_sample_preview(raw_manifest, shard_indices)}",
            flush=True,
        )

        encoder = OrigamiFeatureEncoder(config) if should_build_encoder else None
        missing_feature_progress = tqdm(
            total=global_missing_feature_count,
            desc="Computing missing origami latents",
            disable=runtime.rank != 0 or global_missing_feature_count == 0,
        )
        if runtime.rank == 0:
            _refresh_global_progress_bar(
                missing_feature_progress,
                config.output_manifest,
                runtime.world_size,
            )

        if runtime.rank == 0 and needs_negative_features:
            negative_features = encoder.build_negative_text_features()
            _atomic_torch_save(negative_features, negative_feature_path)
        _barrier(runtime)

        normalized_entries = []
        skipped = []
        processed_since_flush = 0
        processed_since_progress = 0
        local_completed_sample_count = 0
        local_completed_missing_feature_count = 0
        progress_every = max(1, config.progress_every)
        progress = tqdm(
            shard_samples,
            desc=f"Precomputing origami features [rank {runtime.rank}]",
            disable=runtime.rank != 0,
        )
        for manifest_index, sample in progress:
            feature_pt_path = os.path.join(
                config.feature_cache_dir, f"{sample['sample_id']}.pt"
            )
            needs_feature_compute = config.overwrite or not os.path.exists(feature_pt_path)
            try:
                reference_clip_path = choose_reference_clip(sample, config.reference_mode)
                if needs_feature_compute:
                    if encoder is None:
                        raise RuntimeError(
                            "Feature encoder was not initialized for a missing feature shard."
                        )
                    feature_pt, reference_clip_path = encoder.encode_sample(sample)
                    latent_length = feature_pt["latent"].shape[2]
                    valid_latent_length = (latent_length // 4) * 4
                    if valid_latent_length < 4:
                        skipped.append(
                            {
                                "manifest_index": manifest_index,
                                "sample_id": sample["sample_id"],
                                "reason": f"latent_t={latent_length}",
                            }
                        )
                        continue
                    _atomic_torch_save(feature_pt, feature_pt_path)

                normalized_entries.append(
                    {
                        "manifest_index": manifest_index,
                        "row": build_normalized_row(
                            sample=sample,
                            feature_pt_path=feature_pt_path,
                            negative_feature_path=negative_feature_path,
                            reference_clip_path=reference_clip_path,
                            history_keep=config.history_keep,
                        ),
                    }
                )
            except Exception as exc:
                skipped.append(
                    {
                        "manifest_index": manifest_index,
                        "sample_id": sample["sample_id"],
                        "reason": str(exc),
                    }
                )

            processed_since_flush += 1
            processed_since_progress += 1
            local_completed_sample_count += 1
            if needs_feature_compute:
                local_completed_missing_feature_count += 1

            if processed_since_progress >= progress_every:
                _write_rank_progress(
                    config.output_manifest,
                    runtime.rank,
                    assigned_samples=len(shard_samples),
                    completed_samples=local_completed_sample_count,
                    missing_features_total=local_missing_feature_count,
                    missing_features_completed=local_completed_missing_feature_count,
                )
                if runtime.rank == 0:
                    _refresh_global_progress_bar(
                        missing_feature_progress,
                        config.output_manifest,
                        runtime.world_size,
                    )
                processed_since_progress = 0

            if config.save_every > 0 and processed_since_flush >= config.save_every:
                flush_rank_outputs(
                    config.output_manifest,
                    runtime.rank,
                    normalized_entries,
                    skipped,
                )
                processed_since_flush = 0

        flush_rank_outputs(
            config.output_manifest,
            runtime.rank,
            normalized_entries,
            skipped,
        )
        _write_rank_progress(
            config.output_manifest,
            runtime.rank,
            assigned_samples=len(shard_samples),
            completed_samples=local_completed_sample_count,
            missing_features_total=local_missing_feature_count,
            missing_features_completed=local_completed_missing_feature_count,
        )
        _barrier(runtime)

        if runtime.rank == 0:
            _refresh_global_progress_bar(
                missing_feature_progress,
                config.output_manifest,
                runtime.world_size,
            )
            if missing_feature_progress is not None:
                missing_feature_progress.close()
            normalized_count, skipped_count = merge_rank_outputs(
                output_manifest=config.output_manifest,
                feature_cache_dir=config.feature_cache_dir,
                world_size=runtime.world_size,
            )
            if skipped_count:
                skipped_path = os.path.join(config.feature_cache_dir, "skipped_samples.json")
                print(
                    f"Skipped {skipped_count} samples. Details written to {skipped_path}",
                    flush=True,
                )
            print(
                f"Wrote {normalized_count} normalized rows to {config.output_manifest}",
                flush=True,
            )
    finally:
        _destroy_distributed(runtime)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute HY-WorldPlay training features for origami step clips."
    )
    parser.add_argument("--raw-manifest", required=True, type=str)
    parser.add_argument("--output-manifest", required=True, type=str)
    parser.add_argument("--feature-cache-dir", required=True, type=str)
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--target-fps", type=int, default=24)
    parser.add_argument("--history-keep", type=int, default=3)
    parser.add_argument(
        "--reference-mode", type=str, default="latest_past_or_global"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--shard-strategy",
        type=str,
        default="balanced_duration",
        choices=SUPPORTED_SHARD_STRATEGIES,
        help="How to assign manifest rows to distributed ranks.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Flush per-rank shard manifests every N local samples for easier resume/debug.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Write rank progress and refresh the global missing-feature progress bar every N local samples.",
    )
    parser.add_argument(
        "--dist-timeout-seconds",
        type=int,
        default=7200,
        help="Distributed process-group timeout for long-running preprocessing jobs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = OrigamiPrecomputeConfig(
        raw_manifest=args.raw_manifest,
        output_manifest=args.output_manifest,
        feature_cache_dir=args.feature_cache_dir,
        model_path=args.model_path,
        device=args.device,
        height=args.height,
        width=args.width,
        target_fps=args.target_fps,
        history_keep=args.history_keep,
        reference_mode=args.reference_mode,
        overwrite=args.overwrite,
        save_every=args.save_every,
        progress_every=args.progress_every,
        shard_strategy=args.shard_strategy,
        dist_timeout_seconds=args.dist_timeout_seconds,
    )
    build_origami_training_cache(config)


if __name__ == "__main__":
    main()
