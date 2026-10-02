# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

# Allow execution via an absolute script path without requiring callers to
# pre-populate PYTHONPATH with the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UTILS_PATH = Path(__file__).resolve().with_name("origami_precompute_utils.py")
UTILS_SPEC = importlib.util.spec_from_file_location(
    "origami_precompute_utils", UTILS_PATH
)
if UTILS_SPEC is None or UTILS_SPEC.loader is None:
    raise RuntimeError(f"Unable to load origami precompute utils from {UTILS_PATH}")
origami_utils = importlib.util.module_from_spec(UTILS_SPEC)
UTILS_SPEC.loader.exec_module(origami_utils)

atomic_json_dump = origami_utils.atomic_json_dump
build_normalized_row = origami_utils.build_normalized_row
choose_reference_clip = origami_utils.choose_reference_clip
get_shard_dir = origami_utils.get_shard_dir


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_recovered_entries(source_output_manifest: str) -> tuple[dict[int, dict], list[dict]]:
    shard_dir = get_shard_dir(source_output_manifest)
    recovered_by_index: dict[int, dict] = {}
    skipped_entries: list[dict] = []

    if not os.path.isdir(shard_dir):
        return recovered_by_index, skipped_entries

    for name in sorted(os.listdir(shard_dir)):
        shard_path = os.path.join(shard_dir, name)
        if not os.path.isfile(shard_path):
            continue
        if name.startswith("normalized_rank") and name.endswith(".json"):
            for entry in _load_json(shard_path):
                manifest_index = int(entry["manifest_index"])
                recovered_by_index[manifest_index] = entry
        elif name.startswith("skipped_rank") and name.endswith(".json"):
            skipped_entries.extend(_load_json(shard_path))

    return recovered_by_index, skipped_entries


def recover_manifest_rows(
    *,
    raw_manifest_path: str,
    source_output_manifest: str,
    feature_cache_dir: str,
    history_keep: int,
    reference_mode: str,
) -> tuple[list[dict], dict[str, int]]:
    raw_manifest = _load_json(raw_manifest_path)
    recovered_by_index, skipped_entries = _load_recovered_entries(source_output_manifest)
    shard_row_count = len(recovered_by_index)
    cache_row_count = 0

    negative_feature_path = os.path.join(feature_cache_dir, "negative_prompt.pt")
    for manifest_index, sample in enumerate(raw_manifest):
        if manifest_index in recovered_by_index:
            continue

        feature_pt_path = os.path.join(feature_cache_dir, f"{sample['sample_id']}.pt")
        if not os.path.exists(feature_pt_path):
            continue

        recovered_by_index[manifest_index] = {
            "manifest_index": manifest_index,
            "row": build_normalized_row(
                sample=sample,
                feature_pt_path=feature_pt_path,
                negative_feature_path=negative_feature_path,
                reference_clip_path=choose_reference_clip(sample, reference_mode),
                history_keep=history_keep,
            ),
        }
        cache_row_count += 1

    merged_entries = [recovered_by_index[idx] for idx in sorted(recovered_by_index)]
    rows = [entry["row"] for entry in merged_entries]
    stats = {
        "raw_manifest_rows": len(raw_manifest),
        "recovered_rows": len(rows),
        "recovered_from_shards": shard_row_count,
        "recovered_from_cache_scan": cache_row_count,
        "skipped_rows_seen": len(skipped_entries),
        "missing_rows": max(len(raw_manifest) - len(rows), 0),
    }
    return rows, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recover a partial or complete origami normalized manifest from shard JSONs "
            "and any existing feature cache files."
        )
    )
    parser.add_argument("--raw-manifest", required=True, type=str)
    parser.add_argument(
        "--source-output-manifest",
        required=True,
        type=str,
        help="The original output manifest path used by origami_step_precompute.py.",
    )
    parser.add_argument(
        "--write-output-manifest",
        type=str,
        default=None,
        help="Where to write the recovered manifest. Defaults to --source-output-manifest.",
    )
    parser.add_argument("--feature-cache-dir", required=True, type=str)
    parser.add_argument("--history-keep", type=int, default=3)
    parser.add_argument(
        "--reference-mode",
        type=str,
        default="latest_past_or_global",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optionally truncate the recovered manifest to the first N manifest rows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    write_output_manifest = args.write_output_manifest or args.source_output_manifest

    rows, stats = recover_manifest_rows(
        raw_manifest_path=args.raw_manifest,
        source_output_manifest=args.source_output_manifest,
        feature_cache_dir=args.feature_cache_dir,
        history_keep=args.history_keep,
        reference_mode=args.reference_mode,
    )

    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    atomic_json_dump(rows, write_output_manifest)
    print(
        json.dumps(
            {
                **stats,
                "written_rows": len(rows),
                "write_output_manifest": write_output_manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


# python ./trainer/dataset/recover_origami_manifest.py   --raw-manifest data/train_manifest_upto_1000007.json   --source-output-manifest data/origami_train_manifest_normalized.json   --feature-cache-dir data/origami_feature_cache   --write-output-manifest data/origami_train_manifest_recovered_321k.json  --max-rows 321987
