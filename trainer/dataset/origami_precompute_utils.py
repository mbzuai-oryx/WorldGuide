# SPDX-License-Identifier: Apache-2.0
import json
import os
import tempfile
from heapq import heapify, heappop, heappush


SUPPORTED_SHARD_STRATEGIES = ("balanced_duration", "contiguous")


def format_origami_text_prompt(sample: dict) -> str:
    return f"Step {sample['step_number']}: {sample['target_caption']}"


def choose_reference_clip(sample: dict, reference_mode: str) -> str:
    if reference_mode != "latest_past_or_global":
        raise ValueError(f"Unsupported reference mode: {reference_mode}")

    if sample.get("is_first_step") and sample.get("global_clip_path"):
        return sample["global_clip_path"]

    past_clip_paths = sample.get("past_clip_paths") or []
    if past_clip_paths:
        return past_clip_paths[-1]

    return sample["clip_path"]


def get_sample_cost(sample: dict) -> float:
    duration = float(sample.get("duration_sec", 0.0) or 0.0)
    if duration > 0:
        return duration
    return 1.0


def _get_rank_bounds(total_size: int, rank: int, world_size: int) -> tuple[int, int]:
    start = (total_size * rank) // world_size
    end = (total_size * (rank + 1)) // world_size
    return start, end


def build_shard_assignments(
    manifest: list[dict],
    world_size: int,
    shard_strategy: str = "balanced_duration",
) -> list[list[int]]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if shard_strategy not in SUPPORTED_SHARD_STRATEGIES:
        raise ValueError(
            f"Unsupported shard strategy: {shard_strategy}. "
            f"Expected one of {SUPPORTED_SHARD_STRATEGIES}."
        )

    if shard_strategy == "contiguous":
        assignments = []
        for rank in range(world_size):
            start, end = _get_rank_bounds(len(manifest), rank, world_size)
            assignments.append(list(range(start, end)))
        return assignments

    assignments = [[] for _ in range(world_size)]
    rank_costs = [0.0 for _ in range(world_size)]
    heap = [(0.0, rank) for rank in range(world_size)]
    heapify(heap)

    ordered_samples = sorted(
        enumerate(manifest),
        key=lambda item: (-get_sample_cost(item[1]), item[0]),
    )
    for manifest_index, sample in ordered_samples:
        current_cost, rank = heappop(heap)
        sample_cost = get_sample_cost(sample)
        assignments[rank].append(manifest_index)
        rank_costs[rank] = current_cost + sample_cost
        heappush(heap, (rank_costs[rank], rank))

    for indices in assignments:
        indices.sort()
    return assignments


def summarize_assignment_costs(
    manifest: list[dict], assignments: list[list[int]]
) -> list[float]:
    totals = []
    for indices in assignments:
        totals.append(sum(get_sample_cost(manifest[idx]) for idx in indices))
    return totals


def build_normalized_row(
    sample: dict,
    feature_pt_path: str,
    negative_feature_path: str,
    reference_clip_path: str,
    history_keep: int,
) -> dict:
    return {
        "sample_id": sample["sample_id"],
        "video_id": sample["video_id"],
        "task": sample["task"],
        "step_number": sample["step_number"],
        "total_steps": sample["total_steps"],
        "clip_path": sample["clip_path"],
        "reference_clip_path": reference_clip_path,
        "history_last_3": (sample.get("history") or [])[-history_keep:],
        "progress_label": sample["progress_label"],
        "done_label": sample["done_label"],
        "is_first_step": sample["is_first_step"],
        "is_last_step": sample["is_last_step"],
        "feature_pt_path": feature_pt_path,
        "negative_feature_path": negative_feature_path,
    }


def atomic_json_dump(payload, output_path: str) -> None:
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=output_dir, suffix=".tmp"
    ) as fp:
        json.dump(payload, fp, indent=2)
        temp_path = fp.name
    os.replace(temp_path, output_path)


def get_shard_dir(output_manifest: str) -> str:
    return f"{output_manifest}.shards"


def get_rank_manifest_shard_path(output_manifest: str, rank: int) -> str:
    return os.path.join(get_shard_dir(output_manifest), f"normalized_rank{rank:05d}.json")


def get_rank_skipped_shard_path(output_manifest: str, rank: int) -> str:
    return os.path.join(get_shard_dir(output_manifest), f"skipped_rank{rank:05d}.json")


def get_progress_dir(output_manifest: str) -> str:
    return os.path.join(get_shard_dir(output_manifest), "progress")


def get_rank_progress_path(output_manifest: str, rank: int) -> str:
    return os.path.join(get_progress_dir(output_manifest), f"progress_rank{rank:05d}.json")


def flush_rank_outputs(
    output_manifest: str,
    rank: int,
    normalized_entries: list[dict],
    skipped_entries: list[dict],
) -> None:
    atomic_json_dump(
        normalized_entries,
        get_rank_manifest_shard_path(output_manifest, rank),
    )
    atomic_json_dump(
        skipped_entries,
        get_rank_skipped_shard_path(output_manifest, rank),
    )


def merge_rank_outputs(
    output_manifest: str,
    feature_cache_dir: str,
    world_size: int,
) -> tuple[int, int]:
    merged_entries = []
    skipped_entries = []
    for rank in range(world_size):
        shard_manifest_path = get_rank_manifest_shard_path(output_manifest, rank)
        shard_skipped_path = get_rank_skipped_shard_path(output_manifest, rank)
        if os.path.exists(shard_manifest_path):
            with open(shard_manifest_path, "r", encoding="utf-8") as fp:
                merged_entries.extend(json.load(fp))
        if os.path.exists(shard_skipped_path):
            with open(shard_skipped_path, "r", encoding="utf-8") as fp:
                skipped_entries.extend(json.load(fp))

    merged_entries.sort(key=lambda item: item["manifest_index"])
    skipped_entries.sort(key=lambda item: item["manifest_index"])
    normalized_rows = [item["row"] for item in merged_entries]
    atomic_json_dump(normalized_rows, output_manifest)

    skipped_path = os.path.join(feature_cache_dir, "skipped_samples.json")
    if skipped_entries:
        skipped_rows = [
            {"sample_id": item["sample_id"], "reason": item["reason"]}
            for item in skipped_entries
        ]
        atomic_json_dump(skipped_rows, skipped_path)
    elif os.path.exists(skipped_path):
        os.unlink(skipped_path)

    return len(normalized_rows), len(skipped_entries)
