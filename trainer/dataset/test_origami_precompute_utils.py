# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from trainer.dataset.origami_precompute_utils import (
    build_normalized_row,
    build_shard_assignments,
    choose_reference_clip,
    format_origami_text_prompt,
    flush_rank_outputs,
    merge_rank_outputs,
    summarize_assignment_costs,
)


def _make_sample(
    sample_id: str,
    duration_sec: float,
    step_number: int,
    *,
    is_first_step: bool = False,
    history: list[str] | None = None,
    past_clip_paths: list[str] | None = None,
) -> dict:
    return {
        "sample_id": sample_id,
        "video_id": "video-1",
        "task": "origami",
        "step_number": step_number,
        "total_steps": 4,
        "clip_path": f"/clips/{sample_id}.mp4",
        "global_clip_path": "/clips/global.mp4",
        "past_clip_paths": past_clip_paths or [],
        "history": history or [],
        "progress_label": f"progress-{step_number}",
        "done_label": f"done-{step_number}",
        "is_first_step": is_first_step,
        "is_last_step": False,
        "target_caption": f"caption-{sample_id}",
        "duration_sec": duration_sec,
    }


def test_balanced_duration_assignment_is_deterministic_and_balanced():
    manifest = [
        _make_sample(f"sample-{idx}", duration_sec=duration, step_number=idx + 1)
        for idx, duration in enumerate([10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7])
    ]

    assignments_a = build_shard_assignments(
        manifest,
        world_size=4,
        shard_strategy="balanced_duration",
    )
    assignments_b = build_shard_assignments(
        manifest,
        world_size=4,
        shard_strategy="balanced_duration",
    )

    assert assignments_a == assignments_b

    totals = summarize_assignment_costs(manifest, assignments_a)
    assert totals == [34.0, 34.0, 34.0, 34.0]

    covered_indices = sorted(idx for shard in assignments_a for idx in shard)
    assert covered_indices == list(range(len(manifest)))


def test_format_origami_text_prompt_uses_step_and_caption_only():
    sample = _make_sample(
        "sample-0",
        duration_sec=2,
        step_number=3,
        history=["should", "not", "be", "used"],
        past_clip_paths=["/clips/old.mp4"],
    )

    assert format_origami_text_prompt(sample) == "Step 3: caption-sample-0"


def test_merge_rank_outputs_restores_manifest_order_and_writes_skips(tmp_path: Path):
    feature_cache_dir = tmp_path / "features"
    feature_cache_dir.mkdir()
    negative_feature_path = feature_cache_dir / "negative_prompt.pt"
    negative_feature_path.write_bytes(b"neg")

    samples = [
        _make_sample("sample-0", 2, 1, is_first_step=True, history=["a"]),
        _make_sample("sample-1", 8, 2, history=["a", "b"], past_clip_paths=["/clips/sample-0.mp4"]),
        _make_sample("sample-2", 4, 3, history=["a", "b", "c"], past_clip_paths=["/clips/sample-1.mp4"]),
    ]

    for sample in samples:
        (feature_cache_dir / f"{sample['sample_id']}.pt").write_bytes(b"feature")

    output_manifest = tmp_path / "normalized.json"

    rank0_rows = [
        {
            "manifest_index": 2,
            "row": build_normalized_row(
                sample=samples[2],
                feature_pt_path=str(feature_cache_dir / "sample-2.pt"),
                negative_feature_path=str(negative_feature_path),
                reference_clip_path=choose_reference_clip(
                    samples[2], "latest_past_or_global"
                ),
                history_keep=2,
            ),
        }
    ]
    rank1_rows = [
        {
            "manifest_index": 0,
            "row": build_normalized_row(
                sample=samples[0],
                feature_pt_path=str(feature_cache_dir / "sample-0.pt"),
                negative_feature_path=str(negative_feature_path),
                reference_clip_path=choose_reference_clip(
                    samples[0], "latest_past_or_global"
                ),
                history_keep=2,
            ),
        },
        {
            "manifest_index": 1,
            "row": build_normalized_row(
                sample=samples[1],
                feature_pt_path=str(feature_cache_dir / "sample-1.pt"),
                negative_feature_path=str(negative_feature_path),
                reference_clip_path=choose_reference_clip(
                    samples[1], "latest_past_or_global"
                ),
                history_keep=2,
            ),
        },
    ]
    rank1_skipped = [
        {
            "manifest_index": 3,
            "sample_id": "sample-3",
            "reason": "latent_t=0",
        }
    ]

    flush_rank_outputs(str(output_manifest), 0, rank0_rows, [])
    flush_rank_outputs(str(output_manifest), 1, rank1_rows, rank1_skipped)
    flush_rank_outputs(str(output_manifest), 2, [], [])

    normalized_count, skipped_count = merge_rank_outputs(
        output_manifest=str(output_manifest),
        feature_cache_dir=str(feature_cache_dir),
        world_size=3,
    )

    assert normalized_count == 3
    assert skipped_count == 1

    merged_rows = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert [row["sample_id"] for row in merged_rows] == ["sample-0", "sample-1", "sample-2"]
    assert merged_rows[0]["reference_clip_path"] == "/clips/global.mp4"
    assert merged_rows[1]["reference_clip_path"] == "/clips/sample-0.mp4"
    assert merged_rows[2]["history_last_3"] == ["b", "c"]

    skipped_rows = json.loads(
        (feature_cache_dir / "skipped_samples.json").read_text(encoding="utf-8")
    )
    assert skipped_rows == [{"sample_id": "sample-3", "reason": "latent_t=0"}]
