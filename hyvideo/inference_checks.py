"""Lightweight checkpoint and progress-file checks; no model imports."""

import json
from pathlib import Path


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def check_checkpoint_tree(root: Path) -> int:
    """Check indexed shard names and readable files without loading tensors."""
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is unavailable: {root}")
    checked = 0
    for index in root.rglob("*.safetensors.index.json"):
        payload = json.loads(index.read_text(encoding="utf-8"))
        shards = set(payload["weight_map"].values())
        for name in sorted(shards):
            shard = index.parent / name
            try:
                with shard.open("rb") as handle:
                    if len(handle.read(8)) != 8:
                        raise ValueError("empty or truncated checkpoint file")
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Cannot read checkpoint shard {shard} (index {index}): {exc}") from exc
            checked += 1
    return checked


def check_inference_checkpoints(args) -> int:
    root = Path(args.model_path)
    checked = sum(check_checkpoint_tree(root / component) for component in (
        f"transformer/{args.resolution}_i2v", "text_encoder", "vae",
    ))
    if args.action_ckpt:
        with Path(args.action_ckpt).open("rb") as handle:
            if len(handle.read(8)) != 8:
                raise ValueError(f"Empty or truncated action checkpoint: {args.action_ckpt}")
    if getattr(args, "step_source", "qwen_planner") != "manifest" and getattr(args, "planner_model_path", None):
        checked += check_checkpoint_tree(Path(args.planner_model_path))
    return checked
