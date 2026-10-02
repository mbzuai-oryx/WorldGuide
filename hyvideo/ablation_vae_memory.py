"""Optional VAE decode memory scheduling for Qwen ablation workers only.

The installed wrapper belongs to one VAE instance. It changes when transformer
weights occupy GPU memory, without changing VAE decoding or tensor precision.
"""

from functools import wraps
from itertools import chain
from typing import Callable, Optional

import torch


def _transformer_device(transformer) -> torch.device:
    devices = {
        tensor.device
        for tensor in chain(transformer.parameters(), transformer.buffers())
    }
    if len(devices) != 1:
        raise ValueError(
            "VAE decode transformer offload requires transformer parameters and "
            f"buffers on one device; found {sorted(map(str, devices))}."
        )
    device = devices.pop()
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported transformer device for VAE decode offload: {device}")
    return device


def install_vae_decode_transformer_offload(
    pipe, log: Optional[Callable[[str], None]] = None
) -> bool:
    """Temporarily park the resident transformer on CPU during VAE decoding.

    Returns True when installed and False when this VAE is already wrapped.
    The caller must opt in and validate that other offload hooks are disabled.
    PyTorch uses the ``cuda`` device API on ROCm as well.

    Successful decoding restores the transformer to its original device. A
    failed decode leaves it on CPU and propagates the original exception: the
    worker should exit, rather than attempt a second allocation during failure
    cleanup and potentially hide the useful traceback.
    """
    original_decode = pipe.vae.decode
    if getattr(original_decode, "_ablation_transformer_offload", False) is True:
        return False
    transformer = pipe.transformer
    # Fail before the expensive rollout if this is a sharded/meta model.
    _transformer_device(transformer)

    def report(message: str) -> None:
        if log is not None:
            log(f"[ablation-vae-memory] {message}")

    @wraps(original_decode)
    def decode_with_transformer_offload(*args, **kwargs):
        original_device = _transformer_device(transformer)
        if original_device.type == "cpu":
            return original_decode(*args, **kwargs)

        # Keep device-specific cache operations on the worker's actual device.
        # These are blocking transfers: no tensor dtype or weight is changed.
        with torch.cuda.device(original_device):
            report(f"transformer_offload_start device={original_device}")
            torch.cuda.synchronize(original_device)
            transformer.to(device=torch.device("cpu"))
            torch.cuda.synchronize(original_device)
            torch.cuda.empty_cache()
            report("transformer_offload_done; vae_decode_start")

            # Deliberately no finally-based restoration on decode failure.
            decoded = original_decode(*args, **kwargs)
            torch.cuda.synchronize(original_device)
            torch.cuda.empty_cache()
            report(f"vae_decode_done; transformer_restore_start device={original_device}")
            transformer.to(device=original_device)
            torch.cuda.synchronize(original_device)
            report("transformer_restore_done")
            return decoded

    decode_with_transformer_offload._ablation_transformer_offload = True
    pipe.vae.decode = decode_with_transformer_offload
    return True
