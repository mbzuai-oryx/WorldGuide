# SPDX-License-Identifier: Apache-2.0
import json
import random

import pytest

from trainer.dataset.origami_step_dataset import (
    OrigamiStepDataset,
    validate_origami_precomputed_manifest,
)
from trainer.dataset.origami_step_online_dataset import (
    OrigamiStepOnlineDataset,
    build_origami_step_online_dataloader,
)


def test_validate_origami_precomputed_manifest_requires_existing_cache_files(tmp_path):
    feature_pt_path = tmp_path / "sample.pt"
    negative_feature_path = tmp_path / "negative_prompt.pt"
    feature_pt_path.write_bytes(b"feature")
    negative_feature_path.write_bytes(b"negative")

    validate_origami_precomputed_manifest(
        [
            {
                "sample_id": "sample-0",
                "clip_path": "/clips/sample-0.mp4",
                "feature_pt_path": str(feature_pt_path),
                "negative_feature_path": str(negative_feature_path),
            }
        ]
    )

    feature_pt_path.unlink()

    with pytest.raises(FileNotFoundError):
        validate_origami_precomputed_manifest(
            [
                {
                    "sample_id": "sample-0",
                    "clip_path": "/clips/sample-0.mp4",
                    "feature_pt_path": str(feature_pt_path),
                    "negative_feature_path": str(negative_feature_path),
                }
            ]
        )


def test_build_origami_step_online_dataloader_requires_zero_workers(tmp_path):
    manifest_path = tmp_path / "raw_manifest.json"
    manifest_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="dataloader_num_workers 0"):
        build_origami_step_online_dataloader(
            json_path=str(manifest_path),
            causal=True,
            window_frames=24,
            batch_size=1,
            num_data_workers=1,
            prefetch_factor=1,
            drop_last=True,
            drop_first_row=False,
            seed=0,
            cfg_rate=0.1,
            i2v_rate=1.0,
            model_path="/tmp/model",
            height=480,
            width=832,
            target_fps=16,
            reference_mode="latest_past_or_global",
        )


def test_origami_step_dataset_state_dict_restores_rng_and_max_frames():
    dataset = OrigamiStepDataset.__new__(OrigamiStepDataset)
    dataset.rng = random.Random(1234)
    dataset.shared_state = {"max_frames": 32}

    saved_state = dataset.state_dict()
    expected_value = dataset.rng.random()

    dataset.shared_state["max_frames"] = 160
    dataset.rng.random()
    dataset.load_state_dict(saved_state)

    assert dataset.shared_state["max_frames"] == 32
    assert dataset.rng.random() == expected_value


def test_origami_step_online_dataset_state_dict_restores_rng_and_max_frames():
    dataset = OrigamiStepOnlineDataset.__new__(OrigamiStepOnlineDataset)
    dataset.rng = random.Random(5678)
    dataset.shared_state = {"max_frames": 64}

    saved_state = dataset.state_dict()
    expected_value = dataset.rng.random()

    dataset.shared_state["max_frames"] = 128
    dataset.rng.random()
    dataset.load_state_dict(saved_state)

    assert dataset.shared_state["max_frames"] == 64
    assert dataset.rng.random() == expected_value
