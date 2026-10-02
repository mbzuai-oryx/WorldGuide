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
from trainer.dataset.origami_precompute_utils import choose_reference_clip
from trainer.dataset.origami_step_dataset import (
    _build_static_intrinsic,
    _build_static_w2c,
)
from trainer.distributed import get_local_torch_device
from trainer.distributed.parallel_state import (
    get_sp_world_size,
    get_world_rank,
    get_world_size,
)


def validate_origami_raw_manifest(
    json_data: list[dict], reference_mode: str
) -> None:
    missing_fields: list[str] = []
    missing_files: list[str] = []

    for idx, row in enumerate(json_data):
        sample_id = row.get("sample_id", f"row[{idx}]")
        for field_name in ("sample_id", "clip_path", "target_caption", "step_number"):
            if not row.get(field_name):
                missing_fields.append(f"{sample_id}:{field_name}")

        clip_path = row.get("clip_path")
        if clip_path and not os.path.exists(clip_path):
            missing_files.append(clip_path)

        try:
            reference_clip_path = choose_reference_clip(row, reference_mode)
        except Exception as exc:
            missing_fields.append(f"{sample_id}:reference_clip({exc})")
            reference_clip_path = None
        if reference_clip_path and not os.path.exists(reference_clip_path):
            missing_files.append(reference_clip_path)

    if missing_fields:
        preview = ", ".join(missing_fields[:5])
        if len(missing_fields) > 5:
            preview = f"{preview}, ..."
        raise ValueError(
            "Origami raw manifest is missing required fields. "
            f"examples ({len(missing_fields)}): {preview}"
        )

    if missing_files:
        preview = ", ".join(missing_files[:5])
        if len(missing_files) > 5:
            preview = f"{preview}, ..."
        raise FileNotFoundError(
            "Origami raw manifest references missing clip paths. "
            f"missing_paths ({len(missing_files)}): {preview}"
        )


class OrigamiStepOnlineDataset(Dataset):
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
        model_path,
        height,
        width,
        target_fps,
        reference_mode,
        origami_memory_steps=0,
        origami_memory_policy="latest_k",
        origami_memory_blend=0.35,
        origami_memory_mode="mode1",
    ):
        del causal, i2v_rate, origami_memory_steps, origami_memory_policy, origami_memory_blend, origami_memory_mode
        with open(json_path, "r", encoding="utf-8") as fp:
            self.json_data = json.load(fp)
        validate_origami_raw_manifest(self.json_data, reference_mode)

        self.all_length = len(self.json_data)
        self.window_frames = window_frames
        self.cfg_rate = cfg_rate
        self.rng = random.Random(seed)
        self.shared_state = shared_state
        self.device = device
        self.model_path = model_path
        self.height = height
        self.width = width
        self.target_fps = target_fps
        self.reference_mode = reference_mode
        self.encoder = None
        self.negative_features = None

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

    def _get_encoder(self):
        if self.encoder is None:
            from trainer.dataset.origami_step_precompute import (
                OrigamiFeatureEncoder,
                OrigamiPrecomputeConfig,
            )

            self.encoder = OrigamiFeatureEncoder(
                OrigamiPrecomputeConfig(
                    raw_manifest="",
                    output_manifest="",
                    feature_cache_dir="",
                    model_path=self.model_path,
                    device=str(self.device),
                    height=self.height,
                    width=self.width,
                    target_fps=self.target_fps,
                    reference_mode=self.reference_mode,
                )
            )
        return self.encoder

    def _get_negative_text_features(self):
        if self.negative_features is None:
            self.negative_features = self._get_encoder().build_negative_text_features()
        return (
            self.negative_features["prompt_embeds"][0],
            self.negative_features["prompt_mask"][0],
            self.negative_features["byt5_text_states"][0],
            self.negative_features["byt5_text_mask"][0],
        )

    def __getitem__(self, idx):
        while True:
            try:
                json_data = self.json_data[idx]
                feature_pt, _ = self._get_encoder().encode_sample(json_data)

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

                if self.rng.random() < self.cfg_rate:
                    (
                        prompt_embed,
                        prompt_mask,
                        byt5_text_states,
                        byt5_text_mask,
                    ) = self._get_negative_text_features()

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
                print(
                    "error:",
                    exc,
                    self.json_data[idx].get("sample_id"),
                    self.json_data[idx].get("clip_path"),
                    flush=True,
                )
                idx = self.rng.randint(0, self.all_length - 1)
        return batch


def build_origami_step_online_dataloader(
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
    model_path,
    height,
    width,
    target_fps,
    reference_mode,
    origami_memory_steps=0,
    origami_memory_policy="latest_k",
    origami_memory_blend=0.35,
    origami_memory_mode="mode1",
) -> tuple[OrigamiStepOnlineDataset, StatefulDataLoader]:
    if num_data_workers != 0:
        raise ValueError(
            "Origami online feature mode requires --dataloader_num_workers 0 "
            "to avoid spawning duplicate GPU encoder workers."
        )
    if not model_path:
        raise ValueError("model_path is required for origami online feature mode.")

    manager = Manager()
    shared_state = manager.dict()
    shared_state["max_frames"] = window_frames

    dataset = OrigamiStepOnlineDataset(
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
        model_path=model_path,
        height=height,
        width=width,
        target_fps=target_fps,
        reference_mode=reference_mode,
        origami_memory_steps=origami_memory_steps,
        origami_memory_policy=origami_memory_policy,
        origami_memory_blend=origami_memory_blend,
        origami_memory_mode=origami_memory_mode,
    )

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=latent_collate_function,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    return dataset, loader
