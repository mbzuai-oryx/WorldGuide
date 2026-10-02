# SPDX-License-Identifier: Apache-2.0
import json
import os
import random
from multiprocessing import Manager

import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from trainer.dataset.ar_camera_hunyuan_w_mem_dataset import (
    DP_SP_BatchSampler,
    latent_collate_function,
)
from trainer.distributed import get_local_torch_device
from trainer.distributed.parallel_state import (
    get_sp_world_size,
    get_world_rank,
    get_world_size,
)


def _build_static_w2c(latent_length: int) -> torch.Tensor:
    return torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(latent_length, 1, 1)


def _build_static_intrinsic(latent_length: int) -> torch.Tensor:
    intrinsic = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    return intrinsic.unsqueeze(0).repeat(latent_length, 1, 1)


def _summarize_path_issues(label: str, paths: list[str], preview_limit: int = 5) -> str:
    preview = ", ".join(paths[:preview_limit])
    if len(paths) > preview_limit:
        preview = f"{preview}, ..."
    return f"{label} ({len(paths)}): {preview}"


def validate_origami_precomputed_manifest(json_data: list[dict]) -> None:
    missing_fields: list[str] = []
    missing_files: list[str] = []
    seen_feature_paths: set[str] = set()
    seen_negative_paths: set[str] = set()

    for idx, row in enumerate(json_data):
        sample_id = row.get("sample_id", f"row[{idx}]")
        feature_pt_path = row.get("feature_pt_path")
        negative_feature_path = row.get("negative_feature_path")

        for field_name in ("sample_id", "clip_path", "feature_pt_path", "negative_feature_path"):
            if not row.get(field_name):
                missing_fields.append(f"{sample_id}:{field_name}")

        if feature_pt_path and feature_pt_path not in seen_feature_paths:
            seen_feature_paths.add(feature_pt_path)
            if not os.path.exists(feature_pt_path):
                missing_files.append(feature_pt_path)

        if negative_feature_path and negative_feature_path not in seen_negative_paths:
            seen_negative_paths.add(negative_feature_path)
            if not os.path.exists(negative_feature_path):
                missing_files.append(negative_feature_path)

    if missing_fields:
        raise ValueError(
            "Origami precomputed manifest is missing required fields. "
            + _summarize_path_issues("examples", missing_fields)
        )

    if missing_files:
        raise FileNotFoundError(
            "Origami precomputed manifest references missing cache files. "
            + _summarize_path_issues("missing_paths", missing_files)
        )


class OrigamiStepDataset(Dataset):
    def __init__(
        self,
        json_path,
        causal,
        window_frames,
        batch_size,
        cfg_rate,
        i2v_rate,
        drop_last,
        drop_first_row,
        seed,
        device,
        shared_state,
        origami_memory_steps=0,
        origami_memory_policy="latest_k",
        origami_memory_blend=0.35,
        origami_memory_mode="mode1",
    ):
        del causal, i2v_rate, device
        with open(json_path, "r", encoding="utf-8") as fp:
            self.json_data = json.load(fp)
        validate_origami_precomputed_manifest(self.json_data)
        self.all_length = len(self.json_data)
        self.window_frames = window_frames
        self.cfg_rate = cfg_rate
        self.rng = random.Random(seed)
        self.shared_state = shared_state
        self.origami_memory_steps = max(int(origami_memory_steps), 0)
        self.origami_memory_policy = origami_memory_policy
        self.origami_memory_blend = float(origami_memory_blend)
        self.origami_memory_blend = min(max(self.origami_memory_blend, 0.0), 1.0)
        self.origami_memory_mode = origami_memory_mode
        self.memory_debug = os.getenv("MEMORY_DEBUG", "0") == "1"
        self.clip_to_feature_path = {
            row["clip_path"]: row["feature_pt_path"]
            for row in self.json_data
            if row.get("clip_path") and row.get("feature_pt_path")
        }
        self._feature_cache: dict[str, dict] = {}
        self._feature_cache_limit = 512

        self.sampler = DP_SP_BatchSampler(
            batch_size=batch_size,
            dataset_size=self.all_length,
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=drop_last,
            drop_first_row=drop_first_row,
            seed=seed,
        )

    def __len__(self):
        return self.all_length

    def update_max_frames(self, training_step):
        if training_step < 500:
            self.shared_state["max_frames"] = 32
        elif training_step < 1000:
            self.shared_state["max_frames"] = 64
        elif training_step < 2000:
            self.shared_state["max_frames"] = 96
        elif training_step < 3000:
            self.shared_state["max_frames"] = 128
        else:
            self.shared_state["max_frames"] = 160

    def state_dict(self) -> dict:
        return {
            "rng_state": self.rng.getstate(),
            "max_frames": int(self.shared_state["max_frames"]),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "rng_state" in state_dict:
            self.rng.setstate(state_dict["rng_state"])
        if "max_frames" in state_dict:
            self.shared_state["max_frames"] = int(state_dict["max_frames"])

    def _get_negative_text_features(self, json_data, feature_pt):
        negative_feature_path = json_data.get("negative_feature_path")
        if negative_feature_path:
            negative_pt = torch.load(
                negative_feature_path, map_location="cpu", weights_only=True
            )
            return (
                negative_pt["prompt_embeds"][0],
                negative_pt["prompt_mask"][0],
                negative_pt["byt5_text_states"][0],
                negative_pt["byt5_text_mask"][0],
            )

        prompt_embed = torch.zeros_like(feature_pt["prompt_embeds"][0])
        prompt_mask = torch.zeros_like(feature_pt["prompt_mask"][0])
        byt5_text_states = torch.zeros_like(feature_pt["byt5_text_states"][0])
        byt5_text_mask = torch.zeros_like(feature_pt["byt5_text_mask"][0])
        return prompt_embed, prompt_mask, byt5_text_states, byt5_text_mask

    def _load_feature_pt_cached(self, feature_pt_path: str) -> dict | None:
        if feature_pt_path in self._feature_cache:
            return self._feature_cache[feature_pt_path]

        try:
            feature_pt = torch.load(feature_pt_path, map_location="cpu", weights_only=True)
        except Exception:
            return None

        if len(self._feature_cache) >= self._feature_cache_limit:
            first_key = next(iter(self._feature_cache))
            self._feature_cache.pop(first_key, None)
        self._feature_cache[feature_pt_path] = feature_pt
        return feature_pt

    def _select_memory_clip_paths(self, json_data: dict) -> list[str]:
        if self.origami_memory_steps <= 0:
            return []

        past_clip_paths = json_data.get("past_clip_paths") or []
        available = [
            clip_path for clip_path in past_clip_paths
            if clip_path in self.clip_to_feature_path
        ]
        if not available:
            return []

        k = min(self.origami_memory_steps, len(available))
        if self.origami_memory_policy == "random_k":
            return self.rng.sample(available, k=k)
        # latest_k (default)
        return available[-k:]

    def _blend_memory_features(
        self,
        json_data: dict,
        current_latent: torch.Tensor,
        image_cond: torch.Tensor,
        vision_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_clip_paths = self._select_memory_clip_paths(json_data)
        if self.origami_memory_steps <= 0 or self.origami_memory_blend <= 0.0:
            return image_cond, vision_states

        memory_latent_sequences: list[torch.Tensor] = []
        memory_vision_states: list[torch.Tensor] = []
        used_memory_clip_paths: list[str] = []
        past_clip_paths = json_data.get("past_clip_paths") or []
        past_clip_set = set(past_clip_paths)
        current_clip_path = json_data.get("clip_path")

        for clip_path in selected_clip_paths:
            if current_clip_path is not None:
                assert (
                    clip_path != current_clip_path
                ), "Memory source unexpectedly points to current clip instead of a past clip."
            assert (
                clip_path in past_clip_set
            ), "Memory source is not in past_clip_paths."
            feature_pt_path = self.clip_to_feature_path.get(clip_path)
            if not feature_pt_path:
                continue

            memory_feature_pt = self._load_feature_pt_cached(feature_pt_path)
            if memory_feature_pt is None:
                continue

            memory_latent = memory_feature_pt.get("latent")
            memory_vision_state = memory_feature_pt.get("vision_states")

            if memory_latent is None or memory_vision_state is None:
                continue

            memory_latent = memory_latent[0]
            memory_vision_state = memory_vision_state[0]

            if memory_latent.dim() != 4 or memory_latent.shape[1] < 1:
                continue

            if (
                memory_latent.shape[0] != current_latent.shape[0]
                or memory_latent.shape[2] != current_latent.shape[2]
                or memory_latent.shape[3] != current_latent.shape[3]
            ):
                continue
            if memory_vision_state.shape != vision_states.shape:
                continue

            memory_latent_sequences.append(memory_latent)
            memory_vision_states.append(memory_vision_state)
            used_memory_clip_paths.append(clip_path)

        target_hist_frames = max(int(current_latent.shape[1]) - 1, 0)
        if memory_latent_sequences:
            memory_latent_hist = torch.cat(memory_latent_sequences, dim=1)
        else:
            memory_latent_hist = current_latent.new_zeros(
                current_latent.shape[0],
                0,
                current_latent.shape[2],
                current_latent.shape[3],
            )

        if memory_latent_hist.shape[1] > target_hist_frames:
            memory_latent_hist = memory_latent_hist[:, -target_hist_frames:, :, :]
        if memory_latent_hist.shape[1] < target_hist_frames:
            pad_count = target_hist_frames - memory_latent_hist.shape[1]
            pad_latent = current_latent.new_zeros(
                current_latent.shape[0],
                pad_count,
                current_latent.shape[2],
                current_latent.shape[3],
            )
            memory_latent_hist = torch.cat([pad_latent, memory_latent_hist], dim=1)

        target_hist = self.origami_memory_steps
        if len(memory_vision_states) > target_hist:
            memory_vision_states = memory_vision_states[-target_hist:]
        if len(memory_vision_states) < target_hist:
            pad_count = target_hist - len(memory_vision_states)
            pad_vision = torch.zeros_like(vision_states)
            memory_vision_states = [pad_vision] * pad_count + memory_vision_states

        memory_vision_hist = torch.stack(memory_vision_states, dim=0) if memory_vision_states else None

        if self.origami_memory_mode == "mode2":
            image_cond = torch.cat([memory_latent_hist, image_cond], dim=1)
            if memory_vision_hist is not None:
                memory_vision_tokens = memory_vision_hist.reshape(
                    -1, memory_vision_hist.shape[-1]
                )
                vision_states = torch.cat([memory_vision_tokens, vision_states], dim=0)

            if self.memory_debug:
                print(
                    "[memory-debug] dataset using past-latent memory: "
                    f"sample_id={json_data.get('sample_id', 'unknown')}, "
                    f"current_clip={current_clip_path}, "
                    f"used_past_clip_preview={used_memory_clip_paths[:2]}, "
                    f"selected_past_clips={len(used_memory_clip_paths)}, "
                    f"history_frames={int(memory_latent_hist.shape[1])}, "
                    f"source=latent",
                    flush=True,
                )
        return image_cond, vision_states

    def __getitem__(self, idx):
        while True:
            try:
                json_data = self.json_data[idx]
                feature_pt = torch.load(
                    json_data["feature_pt_path"], map_location="cpu", weights_only=True
                )

                latent = feature_pt["latent"][0]
                latent_length = latent.shape[1]
                max_frames = int(self.shared_state["max_frames"]) // 4 * 4
                max_length = min(max_frames, latent_length // 4 * 4)
                if max_length < 4:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue
                

                latent = latent[:, :max_length, ...]
                prompt_embed = feature_pt["prompt_embeds"][0]
                prompt_mask = feature_pt["prompt_mask"][0]
                image_cond = feature_pt["image_cond"][0]
                vision_states = feature_pt["vision_states"][0]
                byt5_text_states = feature_pt["byt5_text_states"][0]
                byt5_text_mask = feature_pt["byt5_text_mask"][0]
                image_cond, vision_states = self._blend_memory_features(
                    json_data, latent, image_cond, vision_states
                )
                if self.origami_memory_mode == "mode2" and self.origami_memory_steps > 0:
                    if image_cond.shape[1] > 1:
                        memory_sum = image_cond[:, :-1].abs().sum().item()
                        if memory_sum == 0.0:
                            idx = self.rng.randint(0, self.all_length - 1)
                            continue

                if self.rng.random() < self.cfg_rate:
                    (
                        prompt_embed,
                        prompt_mask,
                        byt5_text_states,
                        byt5_text_mask,
                    ) = self._get_negative_text_features(json_data, feature_pt)

                latent_t = latent.shape[1]
                w2c = _build_static_w2c(latent_t)
                intrinsic = _build_static_intrinsic(latent_t)
                action = torch.zeros(latent_t, dtype=torch.long)
                i2v_mask = torch.ones_like(latent)

                batch = {
                    "i2v_mask": i2v_mask,
                    "latent": latent,
                    "prompt_embed": prompt_embed,
                    "w2c": w2c,
                    "intrinsic": intrinsic,
                    "action": action,
                    "action_for_pe": action,
                    "context_frames_list": None,
                    "select_window_out_flag": 0,
                    "video_path": json_data["clip_path"],
                    "max_length": max_frames,
                    "image_cond": image_cond,
                    "vision_states": vision_states,
                    "prompt_mask": prompt_mask,
                    "byt5_text_states": byt5_text_states,
                    "byt5_text_mask": byt5_text_mask,
                }
                break
            except Exception as exc:
                print("error:", exc, json_data.get("feature_pt_path"), flush=True)
                idx = self.rng.randint(0, self.all_length - 1)
        return batch


def build_origami_step_dataloader(
    json_path,
    causal,
    window_frames,
    batch_size,
    num_data_workers,
    prefetch_factor,
    drop_last,
    drop_first_row,
    seed,
    cfg_rate,
    i2v_rate,
    origami_memory_steps=0,
    origami_memory_policy="latest_k",
    origami_memory_blend=0.35,
    origami_memory_mode="mode1",
) -> tuple[OrigamiStepDataset, StatefulDataLoader]:
    manager = Manager()
    shared_state = manager.dict()
    shared_state["max_frames"] = window_frames

    dataset = OrigamiStepDataset(
        json_path,
        causal,
        window_frames,
        batch_size,
        cfg_rate,
        i2v_rate,
        drop_last=drop_last,
        drop_first_row=drop_first_row,
        seed=seed,
        device=get_local_torch_device(),
        shared_state=shared_state,
        origami_memory_steps=origami_memory_steps,
        origami_memory_policy=origami_memory_policy,
        origami_memory_blend=origami_memory_blend,
        origami_memory_mode=origami_memory_mode,
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_sampler=dataset.sampler,
        collate_fn=latent_collate_function,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    if num_data_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = StatefulDataLoader(**loader_kwargs)
    return dataset, loader
