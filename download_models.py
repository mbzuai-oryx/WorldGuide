#!/usr/bin/env python3
"""
Download all checkpoints required for WorldGuide inference into ./ckpts.

Usage:
    python download_models.py --hf_token <your_token>

Resulting layout:
    ckpts/
    ├── WorldGuide-Ckpt/                  # MBZUAI/WorldGuide-Ckpt
    │   ├── transformer/                  # WorldGuide video DiT (config.json + weights)
    │   └── text_encoder/llm/             # Fine-tuned Qwen2.5-VL-7B: DiT text encoder + ContextPlanner
    └── HunyuanVideo-1.5/                 # --model_path
        ├── vae/  scheduler/              # tencent/HunyuanVideo-1.5
        ├── text_encoder/
        │   ├── llm -> ../../WorldGuide-Ckpt/text_encoder/llm
        │   ├── byt5-small/               # google/byt5-small
        │   └── Glyph-SDXL-v2/            # AI-ModelScope/Glyph-SDXL-v2 (ModelScope)
        └── vision_encoder/siglip/        # black-forest-labs/FLUX.1-Redux-dev (gated)

The HF token is required for the gated vision encoder.
Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
"""

import argparse
import os
import shutil
import sys

WORLDGUIDE_REPO = "MBZUAI/WorldGuide-Ckpt"
HUNYUAN_REPO = "tencent/HunyuanVideo-1.5"


def check_dependencies():
    """Check and install required dependencies."""
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install -U 'huggingface_hub[cli]'")

    try:
        import modelscope  # noqa: F401
    except ImportError:
        print("Installing modelscope...")
        os.system("pip install modelscope")


def header(step, title):
    print("\n" + "=" * 60)
    print(f"[{step}] {title}")
    print("=" * 60)


def is_populated(path, min_entries=1):
    return os.path.isdir(path) and len(os.listdir(path)) >= min_entries


def download_worldguide(ckpt_dir, hf_token):
    """Download the WorldGuide DiT and the fine-tuned Qwen text encoder / planner."""
    from huggingface_hub import snapshot_download

    header("1/5", f"Downloading {WORLDGUIDE_REPO} (DiT + text encoder/planner)...")
    target = os.path.join(ckpt_dir, "WorldGuide-Ckpt")
    snapshot_download(WORLDGUIDE_REPO, local_dir=target, token=hf_token)
    print(f"Downloaded to: {target}")
    return target


def download_hunyuan_video(hunyuan_dir):
    """Download HunyuanVideo-1.5 VAE and scheduler. The base DiT is not needed:
    the WorldGuide checkpoint contains all of its weights."""
    from huggingface_hub import snapshot_download

    header("2/5", f"Downloading {HUNYUAN_REPO} (vae, scheduler)...")
    snapshot_download(
        HUNYUAN_REPO, local_dir=hunyuan_dir, allow_patterns=["vae/*", "scheduler/*"]
    )
    print(f"Downloaded to: {hunyuan_dir}")


def link_text_encoder(hunyuan_dir, worldguide_dir):
    """Expose the WorldGuide text encoder at HunyuanVideo-1.5/text_encoder/llm."""
    text_encoder_base = os.path.join(hunyuan_dir, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)
    llm_target = os.path.join(text_encoder_base, "llm")
    llm_source = os.path.join(worldguide_dir, "text_encoder", "llm")

    if os.path.islink(llm_target):
        os.unlink(llm_target)
    elif os.path.exists(llm_target):
        shutil.rmtree(llm_target)
    os.symlink(os.path.relpath(llm_source, text_encoder_base), llm_target)
    print(f"Linked {llm_target} -> {llm_source}")


def download_byt5_encoders(hunyuan_dir):
    """Download ByT5 text encoders (byt5-small and Glyph-SDXL-v2)."""
    from huggingface_hub import snapshot_download
    from modelscope import snapshot_download as ms_snapshot_download

    header("3/5", "Downloading ByT5 text encoders...")
    text_encoder_base = os.path.join(hunyuan_dir, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)

    byt5_target = os.path.join(text_encoder_base, "byt5-small")
    if is_populated(byt5_target, 3):
        print(f"byt5-small already exists at: {byt5_target}")
    else:
        snapshot_download(
            "google/byt5-small",
            local_dir=byt5_target,
            ignore_patterns=["*.h5", "*.msgpack"],
        )
        print(f"Downloaded to: {byt5_target}")

    glyph_target = os.path.join(text_encoder_base, "Glyph-SDXL-v2")
    if os.path.exists(os.path.join(glyph_target, "checkpoints", "byt5_model.pt")):
        print(f"Glyph-SDXL-v2 already exists at: {glyph_target}")
        return
    if os.path.exists(glyph_target):
        shutil.rmtree(glyph_target)

    print("Downloading AI-ModelScope/Glyph-SDXL-v2 from ModelScope...")
    glyph_cache = ms_snapshot_download(
        "AI-ModelScope/Glyph-SDXL-v2", cache_dir="/tmp/glyph_cache"
    )
    shutil.copytree(glyph_cache, glyph_target)
    print(f"Copied to: {glyph_target}")


def download_vision_encoder(hunyuan_dir, hf_token):
    """Download SigLIP vision encoder from FLUX.1-Redux-dev."""
    from huggingface_hub import snapshot_download

    header("4/5", "Downloading vision encoder (SigLIP from FLUX.1-Redux-dev)...")
    siglip_target = os.path.join(hunyuan_dir, "vision_encoder", "siglip")
    if is_populated(siglip_target, 3):
        print(f"siglip already exists at: {siglip_target}")
        return
    if not hf_token:
        print("WARNING: No HF token provided; skipping the gated vision encoder.")
        print("Request access at https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev")
        return

    try:
        snapshot_download(
            "black-forest-labs/FLUX.1-Redux-dev", local_dir=siglip_target, token=hf_token
        )
        print(f"Downloaded to: {siglip_target}")
    except Exception as e:
        print(f"ERROR: Failed to download vision encoder: {e}")
        print("Make sure you have access to FLUX.1-Redux-dev and your token is valid.")


def verify(ckpt_dir, hunyuan_dir):
    header("5/5", "Verifying downloads...")
    required = [
        os.path.join(ckpt_dir, "WorldGuide-Ckpt", "transformer", "config.json"),
        os.path.join(ckpt_dir, "WorldGuide-Ckpt", "transformer", "diffusion_pytorch_model.safetensors"),
        os.path.join(hunyuan_dir, "text_encoder", "llm", "config.json"),
        os.path.join(hunyuan_dir, "text_encoder", "byt5-small"),
        os.path.join(hunyuan_dir, "text_encoder", "Glyph-SDXL-v2", "checkpoints", "byt5_model.pt"),
        os.path.join(hunyuan_dir, "vision_encoder", "siglip", "image_encoder"),
        os.path.join(hunyuan_dir, "vae"),
        os.path.join(hunyuan_dir, "scheduler"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    for path in missing:
        print(f"MISSING: {path}")
    if missing:
        sys.exit(1)
    print("All checkpoints are in place. Inference and training scripts use the default paths under ./ckpts.")


def main():
    parser = argparse.ArgumentParser(
        description="Download all required checkpoints for WorldGuide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python download_models.py --hf_token hf_xxxxxxxxxxxxx

Note:
    The HuggingFace token is required for the gated vision encoder
    (black-forest-labs/FLUX.1-Redux-dev). You need to:
    1. Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
    2. Create a token at: https://huggingface.co/settings/tokens (select "Read" permission)
        """,
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (required for the gated vision encoder; defaults to $HF_TOKEN)",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="ckpts", help="Output directory for all checkpoints"
    )
    parser.add_argument(
        "--skip_vision_encoder",
        action="store_true",
        help="Skip downloading the vision encoder (if you don't have FLUX access yet)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("WorldGuide Model Download Script")
    print("=" * 60)

    check_dependencies()

    hunyuan_dir = os.path.join(args.ckpt_dir, "HunyuanVideo-1.5")
    worldguide_dir = download_worldguide(args.ckpt_dir, args.hf_token)
    download_hunyuan_video(hunyuan_dir)
    link_text_encoder(hunyuan_dir, worldguide_dir)
    download_byt5_encoders(hunyuan_dir)
    if args.skip_vision_encoder:
        header("4/5", "Skipping vision encoder download (--skip_vision_encoder flag)")
    else:
        download_vision_encoder(hunyuan_dir, args.hf_token)
    verify(args.ckpt_dir, hunyuan_dir)


if __name__ == "__main__":
    main()
