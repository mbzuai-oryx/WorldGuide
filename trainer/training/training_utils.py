# SPDX-License-Identifier: Apache-2.0
import gc
import shutil
import json
import math
import os
import time
from collections.abc import Callable, Iterator
from enum import Enum
from typing import Any

import torch
import torch.distributed as dist
from einops import rearrange
from safetensors.torch import save_file
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from trainer.distributed.parallel_state import (get_sp_parallel_rank,
                                                  get_sp_world_size)
from trainer.logger import init_logger
from trainer.training.checkpointing_utils import ModelWrapper

logger = init_logger(__name__)

_HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES = False
TRAINING_STATE_FILENAME = "training_state.pt"


def _get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _cleanup_resume_tensors(rank: int) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        logger.info(
            "rank: %s, CUDA memory after resume cleanup: allocated=%.2f MB reserved=%.2f MB",
            rank,
            torch.cuda.memory_allocated() / 1024**2,
            torch.cuda.memory_reserved() / 1024**2,
            local_main_process_only=False,
        )


def gather_state_dict_on_cpu_rank0(
    model,
    device: torch.device | None = None,
) -> dict[str, Any]:
    rank = _get_rank()
    cpu_state_dict = {}
    sharded_sd = model.state_dict()
    param_requires_grad = set([
        k.replace("._checkpoint_wrapped_module.", ".")
        for k, v in dict(model.named_parameters()).items() if v.requires_grad
    ])
    for param_name, param in sharded_sd.items():
        if param_name not in param_requires_grad:
            continue
        if hasattr(param, "_local_tensor"):
            # DTensor case
            if param.is_cpu:
                # Gather directly on CPU
                param = param.full_tensor()
            else:
                if device is not None:
                    param = param.to(device)
                param = param.full_tensor()
        else:
            # Regular tensor case
            if param.is_cpu:
                pass
            else:
                if device is not None:
                    param = param.to(device)

        if rank == 0:
            cpu_state_dict[param_name] = param.cpu()

    return cpu_state_dict

def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator,
    logit_mean: float | None = None,
    logit_std: float | None = None,
    mode_scale: float | None = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size, ),
            device="cpu",
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2)**2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
    return u


def get_sigmas(noise_scheduler,
               device,
               timesteps,
               n_dim=4,
               dtype=torch.float32) -> torch.Tensor:
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item()
                    for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def _list_checkpoint_dirs(output_dir: str) -> list[tuple[int, str]]:
    checkpoint_dirs: list[tuple[int, str]] = []
    if not os.path.exists(output_dir):
        return checkpoint_dirs

    for entry in os.listdir(output_dir):
        if not entry.startswith("checkpoint-"):
            continue
        step_str = entry.split("checkpoint-")[-1]
        try:
            step = int(step_str)
        except ValueError:
            continue
        checkpoint_dirs.append((step, os.path.join(output_dir, entry)))

    checkpoint_dirs.sort(key=lambda item: item[0])
    return checkpoint_dirs


def prune_old_checkpoints(output_dir: str, rank: int, save_limit: int) -> None:
    if rank != 0 or save_limit <= 0:
        return

    checkpoint_dirs = _list_checkpoint_dirs(output_dir)
    if len(checkpoint_dirs) <= save_limit:
        return

    stale_checkpoints = checkpoint_dirs[:-save_limit]
    for stale_step, stale_dir in stale_checkpoints:
        logger.info("Removing old checkpoint checkpoint-%s at %s",
                    stale_step,
                    stale_dir)
        shutil.rmtree(stale_dir, ignore_errors=True)


def resolve_training_schedule(
    max_train_steps: int | None,
    num_train_epochs: int | None,
    num_update_steps_per_epoch: int,
) -> tuple[int, int]:
    if num_update_steps_per_epoch <= 0:
        raise ValueError("num_update_steps_per_epoch must be positive.")

    max_train_steps = 0 if max_train_steps is None else max_train_steps
    num_train_epochs = 0 if num_train_epochs is None else num_train_epochs

    if max_train_steps > 0:
        resolved_max_train_steps = max_train_steps
        resolved_num_train_epochs = math.ceil(
            resolved_max_train_steps / num_update_steps_per_epoch
        )
    elif num_train_epochs > 0:
        resolved_num_train_epochs = num_train_epochs
        resolved_max_train_steps = (
            resolved_num_train_epochs * num_update_steps_per_epoch
        )
    else:
        raise ValueError(
            "Either max_train_steps or num_train_epochs must be greater than 0."
        )

    return resolved_max_train_steps, resolved_num_train_epochs


def load_checkpoint_metadata(checkpoint_path: str) -> dict[str, Any] | None:
    metadata_path = os.path.join(checkpoint_path, "checkpoint_metadata.json")
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def _build_rank_training_state(
    transformer,
    optimizer=None,
    dataloader=None,
    scheduler=None,
    noise_generator=None,
    noise_gen_cuda=None,
    validation_random_generator=None,
    current_epoch: int = 0,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    use_full_state_dict = dist.is_initialized()
    state: dict[str, Any] = {
        "step": step,
        "current_epoch": current_epoch,
        "world_size": _get_world_size(),
        "model": ModelWrapper(
            transformer,
            cpu_offload=True,
            full_state_dict=use_full_state_dict,
        ).state_dict(),
    }

    if dataloader is not None:
        state["dataloader_state"] = dataloader.state_dict()

    dataset = getattr(dataloader, "dataset", None) if dataloader is not None else None
    if dataset is not None and hasattr(dataset, "state_dict"):
        state["dataset_state"] = dataset.state_dict()

    if metadata is not None:
        state["checkpoint_metadata"] = metadata
        for key in (
            "train_batch_size",
            "train_sp_batch_size",
            "gradient_accumulation_steps",
            "sp_size",
            "total_batch_size",
            "samples_seen_global",
        ):
            if key in metadata:
                state[key] = metadata[key]

    return state


def _save_training_state(
    transformer,
    rank: int,
    save_dir: str,
    step: int,
    optimizer=None,
    dataloader=None,
    scheduler=None,
    noise_generator=None,
    noise_gen_cuda=None,
    validation_random_generator=None,
    current_epoch: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    training_state_path = os.path.join(save_dir, TRAINING_STATE_FILENAME)
    if rank == 0:
        logger.info("Saving global training state to %s", training_state_path)
    begin_time = time.perf_counter()
    training_state = _build_rank_training_state(
        transformer,
        optimizer=optimizer,
        dataloader=dataloader,
        scheduler=scheduler,
        noise_generator=noise_generator,
        noise_gen_cuda=noise_gen_cuda,
        validation_random_generator=validation_random_generator,
        current_epoch=current_epoch,
        step=step,
        metadata=metadata,
    )
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        for entry in os.listdir(save_dir):
            if entry.startswith("training_state_rank") and entry.endswith(".pt"):
                os.remove(os.path.join(save_dir, entry))
        legacy_distributed_dir = os.path.join(save_dir, "distributed_checkpoint")
        if os.path.isdir(legacy_distributed_dir):
            shutil.rmtree(legacy_distributed_dir, ignore_errors=True)
        torch.save(training_state, training_state_path)
        end_time = time.perf_counter()
        logger.info("Global training state saved in %.2f seconds",
                    end_time - begin_time)

    if dist.is_initialized():
        dist.barrier()


def _load_training_state(
    transformer,
    rank: int,
    checkpoint_path: str,
    optimizer=None,
    dataloader=None,
    scheduler=None,
    noise_generator=None,
    noise_gen_cuda=None,
    validation_random_generator=None,
    reset_dataset: bool = False,
) -> dict[str, int] | None:
    training_state_path = os.path.join(checkpoint_path, TRAINING_STATE_FILENAME)
    if not os.path.exists(training_state_path):
        return None

    load_full_training_state_on_this_rank = rank == 0

    training_state: dict[str, Any]
    if load_full_training_state_on_this_rank:
        logger.info("Loading global training state from %s", training_state_path)
        training_state = torch.load(
            training_state_path,
            map_location="cpu",
            weights_only=False,
        )
    else:
        training_state = {}

    expected_world_size = int(
        training_state.get("world_size", _get_world_size())
        if rank == 0 else _get_world_size()
    )
    if dist.is_initialized():
        expected_world_size_list = [expected_world_size]
        dist.broadcast_object_list(expected_world_size_list, src=0)
        expected_world_size = int(expected_world_size_list[0])
    world_size = _get_world_size()
    if expected_world_size != world_size:
        raise ValueError(
            f"Checkpoint {checkpoint_path} expects world_size={expected_world_size}, "
            f"but current run has world_size={world_size}."
        )

    begin_time = time.perf_counter()
    use_full_state_dict = dist.is_initialized()
    model_state = training_state.pop("model", {})
    ModelWrapper(
        transformer,
        cpu_offload=dist.is_initialized(),
        full_state_dict=use_full_state_dict,
        broadcast_from_rank0=dist.is_initialized(),
    ).load_state_dict(model_state if rank == 0 else {})
    del model_state
    _cleanup_resume_tensors(rank)
    if dist.is_initialized():
        dist.barrier()

    aux_state = None
    if rank == 0:
        aux_state = {
            "step": int(training_state.pop("step", 0)),
            "current_epoch": int(training_state.pop("current_epoch", 0)),
            "dataset_state": training_state.pop("dataset_state", None),
            "dataloader_state": training_state.pop("dataloader_state", None),
        }
    training_state.clear()

    if dist.is_initialized():
        aux_state_list = [aux_state]
        dist.broadcast_object_list(aux_state_list, src=0)
        aux_state = aux_state_list[0]

    if aux_state is None:
        aux_state = {
            "step": 0,
            "current_epoch": 0,
        }

    if not reset_dataset:
        dataset = getattr(dataloader, "dataset", None) if dataloader is not None else None
        if (
            dataset is not None
            and aux_state.get("dataset_state") is not None
            and hasattr(dataset, "load_state_dict")
        ):
            dataset.load_state_dict(aux_state["dataset_state"])

        if dataloader is not None and aux_state.get("dataloader_state") is not None:
            dataloader.load_state_dict(aux_state["dataloader_state"])

    resume_info = {
        "step": int(aux_state.get("step", 0)),
        "current_epoch": int(aux_state.get("current_epoch", 0)),
    }
    del training_state
    del aux_state
    if dist.is_initialized():
        del aux_state_list
    _cleanup_resume_tensors(rank)
    if dist.is_initialized():
        dist.barrier()

    end_time = time.perf_counter()
    if rank == 0:
        logger.info("Global training state loaded in %.2f seconds",
                    end_time - begin_time)

    return resume_info


def save_checkpoint(transformer,
                    rank,
                    output_dir,
                    step,
                    optimizer=None,
                    dataloader=None,
                    scheduler=None,
                    noise_generator=None,
                    noise_gen_cuda=None,
                    validation_random_generator=None,
                    current_epoch: int = 0,
                    save_training_state: bool = True,
                    save_consolidated_checkpoint: bool = True,
                    checkpoints_total_limit: int = 0,
                    metadata: dict[str, Any] | None = None) -> None:
    """
    Save a checkpoint containing resumable training state and/or consolidated weights.
    """
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)

    if save_training_state:
        _save_training_state(
            transformer,
            rank,
            save_dir,
            step,
            optimizer=optimizer,
            dataloader=dataloader,
            scheduler=scheduler,
            noise_generator=noise_generator,
            noise_gen_cuda=noise_gen_cuda,
            validation_random_generator=validation_random_generator,
            current_epoch=current_epoch,
            metadata=metadata,
        )

    if save_consolidated_checkpoint:
        cpu_state = gather_state_dict_on_cpu_rank0(transformer, device=None)
        if rank == 0:
            # Save model weights (consolidated)
            transformer_save_dir = os.path.join(save_dir, "transformer")
            os.makedirs(transformer_save_dir, exist_ok=True)
            weight_path = os.path.join(transformer_save_dir,
                                       "diffusion_pytorch_model.safetensors")
            logger.info("rank: %s, saving consolidated checkpoint to %s",
                        rank,
                        weight_path,
                        local_main_process_only=False)

            # Convert training format to diffusers format and save
            # diffusers_state_dict = custom_to_hf_state_dict(
            #     cpu_state, transformer.reverse_param_names_mapping)
            save_file(cpu_state, weight_path)

            logger.info("rank: %s, consolidated checkpoint saved to %s",
                        rank,
                        weight_path,
                        local_main_process_only=False)

            # Save model config
            config_dict = transformer.config
            if "dtype" in config_dict:
                del config_dict["dtype"]  # TODO
            config_path = os.path.join(transformer_save_dir, "config.json")
            # save dict as json
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=4)
            logger.info("--> checkpoint saved at step %s to %s", step, weight_path)

        # Keep non-zero ranks from racing back into training while rank 0 is still
        # finalizing the consolidated checkpoint files.
        if dist.is_initialized():
            dist.barrier()

    if rank == 0 and metadata is not None:
        metadata_path = os.path.join(save_dir, "checkpoint_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        logger.info("Saved checkpoint metadata to %s", metadata_path)

    prune_old_checkpoints(output_dir, rank, checkpoints_total_limit)
    if dist.is_initialized():
        dist.barrier()

def load_checkpoint(transformer,
                    rank,
                    checkpoint_path,
                    optimizer=None,
                    dataloader=None,
                    scheduler=None,
                    noise_generator=None,
                    noise_gen_cuda=None,
                    validation_random_generator=None,
                    reset_dataset: bool = False) -> dict[str, int]:
    """
    Load a checkpoint and return resume information.
    """
    if not os.path.exists(checkpoint_path):
        logger.warning("Checkpoint path %s does not exist", checkpoint_path)
        return {"step": 0, "current_epoch": 0}

    # Extract step number from checkpoint path
    step = int(os.path.basename(checkpoint_path).split('-')[-1])

    if rank == 0:
        logger.info("Loading checkpoint from step %s", step)

    loaded_state = _load_training_state(
        transformer,
        rank,
        checkpoint_path,
        optimizer=optimizer,
        dataloader=dataloader,
        scheduler=scheduler,
        noise_generator=noise_generator,
        noise_gen_cuda=noise_gen_cuda,
        validation_random_generator=validation_random_generator,
        reset_dataset=reset_dataset,
    )
    if loaded_state is not None:
        logger.info("--> checkpoint loaded from step %s", loaded_state["step"])
        return loaded_state

    metadata = load_checkpoint_metadata(checkpoint_path)
    if metadata is not None:
        logger.warning(
            "Checkpoint %s is missing %s and cannot resume the exact dataloader/sample index. Metadata: %s",
            checkpoint_path,
            TRAINING_STATE_FILENAME,
            metadata,
        )
    else:
        logger.warning(
            "Checkpoint %s is missing %s and cannot be resumed exactly.",
            checkpoint_path,
            TRAINING_STATE_FILENAME,
        )
    return {"step": 0, "current_epoch": 0}


def normalize_dit_input(model_type, latents, vae) -> torch.Tensor:
    if model_type == "hunyuan_hf" or model_type == "hunyuan":
        return latents * 0.476986
    elif model_type == "wan":
        latents_mean = torch.tensor(vae.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.latents_std)

        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents
    else:
        raise NotImplementedError(f"model_type {model_type} not supported")


def shard_latents_across_sp(latents: torch.Tensor,
                            num_latent_t: int) -> torch.Tensor:
    sp_world_size = get_sp_world_size()
    rank_in_sp_group = get_sp_parallel_rank()
    latents = latents[:, :, :num_latent_t]
    if sp_world_size > 1:
        latents = rearrange(latents,
                            "b c (n s) h w -> b c n s h w",
                            n=sp_world_size).contiguous()
        latents = latents[:, :, rank_in_sp_group, :, :, :]
    return latents

def shard_latents_dim_across_sp(latents: torch.Tensor, total_dim) -> torch.Tensor:
    sp_world_size = get_sp_world_size()
    rank_in_sp_group = get_sp_parallel_rank()
    # latents = latents[:, :, :num_latent_t]
    if sp_world_size > 1:
        if total_dim == 5:
            latents = rearrange(latents,
                                "b c (n s) h w -> b c n s h w",
                                n=sp_world_size).contiguous()
            latents = latents[:, :, rank_in_sp_group, :, :, :]
        elif total_dim == 4:
            latents = rearrange(latents,
                                "b (n s) h w -> b n s h w",
                                n=sp_world_size).contiguous()
            latents = latents[:, rank_in_sp_group, :, :, :]
        elif total_dim == 3:
            latents = rearrange(latents,
                                "b (n s) c -> b n s c",
                                n=sp_world_size).contiguous()
            latents = latents[:, rank_in_sp_group, :, :]
        elif total_dim == 2:
            latents = rearrange(latents,
                                "b (n s)-> b n s",
                                n=sp_world_size).contiguous()
            latents = latents[:, rank_in_sp_group, :]
        elif total_dim == 1:
            latents = rearrange(latents,
                                "(n s)-> n s",
                                n=sp_world_size).contiguous()
            latents = latents[rank_in_sp_group, :]
        else:
            raise NotImplementedError(f"total_dim {total_dim} not supported")
    return latents


def clip_grad_norm_while_handling_failing_dtensor_cases(
    parameters: torch.Tensor | list[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: torch.distributed.device_mesh.DeviceMesh | None = None,
) -> torch.Tensor | None:
    global _HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES

    if not _HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES:
        try:
            return clip_grad_norm_(parameters, max_norm, norm_type,
                                   error_if_nonfinite, foreach, pp_mesh)
        except NotImplementedError as e:
            if "DTensor does not support cross-mesh operation" in str(e):
                # https://github.com/pytorch/pytorch/issues/134212
                logger.warning(
                    "DTensor does not support cross-mesh operation. If you haven't fully tensor-parallelized your "
                    "model, while combining other parallelisms such as FSDP, it could be the reason for this error. "
                    "Gradient clipping will be skipped and gradient norm will not be logged."
                )
        except Exception as e:
            logger.warning(
                "An error occurred while clipping gradients: %s. Gradient clipping will be skipped and gradient "
                "norm will not be logged.", e)
            _HAS_ERRORED_CLIP_GRAD_NORM_WHILE_HANDLING_FAILING_DTENSOR_CASES = True
    return None


# Copied from https://github.com/pytorch/torchtitan/blob/4a169701555ab9bd6ca3769f9650ae3386b84c6e/torchtitan/utils.py#L362
@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | list[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: torch.distributed.device_mesh.DeviceMesh | None = None,
) -> torch.Tensor:
    r"""
    Clip the gradient norm of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters (`torch.Tensor` or `List[torch.Tensor]`):
            Tensors that will have gradients normalized.
        max_norm (`float`):
            Maximum norm of the gradients after clipping.
        norm_type (`float`, defaults to `2.0`):
            Type of p-norm to use. Can be `inf` for infinity norm.
        error_if_nonfinite (`bool`, defaults to `False`):
            If `True`, an error is thrown if the total norm of the gradients 
            from `parameters` is `nan`, `inf`, or `-inf`.
        foreach (`bool`, defaults to `None`):
            Use the faster foreach-based implementation. If `None`, 
            use the foreach implementation for CUDA and CPU native tensors
            and silently fall back to the slow implementation for other device types.
        pp_mesh (`torch.distributed.device_mesh.DeviceMesh`, defaults to `None`):
            Pipeline parallel device mesh. If not `None`, will reduce gradient norm across PP stages.

    Returns:
        `torch.Tensor`:
            Total norm of the gradients
    """
    grads = [p.grad for p in parameters if p.grad is not None]

    # TODO(aryan): Wait for next Pytorch release to use `torch.nn.utils.get_total_norm`
    # total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # It has two purposes:
    #   1. to make sure the total norm is computed correctly when PP is used (see below)
    #   2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, torch.distributed.tensor.DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        raise NotImplementedError("Pipeline parallel is not supported")
        if math.isinf(norm_type):
            dist.all_reduce(total_norm,
                            op=dist.ReduceOp.MAX,
                            group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm,
                            op=dist.ReduceOp.SUM,
                            group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


@torch.no_grad()
def _clip_grads_with_norm_(
    parameters: torch.Tensor | list[torch.Tensor],
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: bool | None = None,
) -> None:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    if len(grads) == 0:
        return
    grouped_grads: dict[tuple[torch.device, torch.dtype],
                        tuple[list[list[torch.Tensor]],
                              list[int]]] = (_group_tensors_by_device_and_dtype(
                                  [grads]))  # type: ignore[assignment]

    clip_coef = max_norm / (total_norm + 1e-6)

    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
                foreach and _device_has_foreach_support(device)):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)


def _get_total_norm(
    tensors: torch.Tensor | list[torch.Tensor],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    tensors = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: dict[tuple[torch.device, torch.dtype],
                          tuple[list[list[torch.Tensor]], list[int]]] = (
                              _group_tensors_by_device_and_dtype(
                                  [tensors]  # type: ignore[list-item]
                              ))  # type: ignore[assignment]

    norms: list[torch.Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        local_tensors = [
            t.to_local()
            if isinstance(t, torch.distributed.tensor.DTensor) else t
            for t in device_tensors
        ]
        if (foreach is None and _has_foreach_support(local_tensors, device)
            ) or (foreach and _device_has_foreach_support(device)):
            norms.extend(torch._foreach_norm(local_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in local_tensors])

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(),
                                               total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`")
    return total_norm


def _get_foreach_kernels_supported_devices() -> list[str]:
    r"""Return the device type list that supports foreach kernels."""
    return ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]


@torch.no_grad()
def _group_tensors_by_device_and_dtype(
    tensorlistlist: list[list[torch.Tensor | None]],
    with_indices: bool = False,
) -> dict[tuple[torch.device, torch.dtype], tuple[
        list[list[torch.Tensor | None]], list[int]]]:
    return torch._C._group_tensors_by_device_and_dtype(  # type: ignore[no-any-return]
        tensorlistlist, with_indices)


def _device_has_foreach_support(device: torch.device) -> bool:
    return device.type in (_get_foreach_kernels_supported_devices() +
                           ["cpu"]) and not torch.jit.is_scripting()


def _has_foreach_support(tensors: list[torch.Tensor],
                         device: torch.device) -> bool:
    return _device_has_foreach_support(device) and all(
        t is None or type(t) in [torch.Tensor] for t in tensors)


def custom_to_hf_state_dict(
    state_dict: dict[str, Any] | Iterator[tuple[str, torch.Tensor]],
    reverse_param_names_mapping: dict[str, tuple[str, int,
                                                 int]]) -> dict[str, Any]:
    """
    Convert trainer's custom model format to diffusers format using reverse_param_names_mapping.
    
    Args:
        state_dict: State dict in trainer's custom format
        reverse_param_names_mapping: Reverse mapping from trainer's custom format to diffusers format
        
    Returns:
        State dict in diffusers format
    """
    assert len(
        reverse_param_names_mapping) > 0, "reverse_param_names_mapping is empty"
    if isinstance(state_dict, Iterator):
        state_dict = dict(state_dict)
    new_state_dict = {}
    # Group parameters that need to be split (merged parameters)
    merge_groups: dict[str, list[tuple[str, int, int]]] = {}

    # First pass: collect all merge groups
    for training_key, (
            diffusers_key, merge_index,
            num_params_to_merge) in reverse_param_names_mapping.items():
        if merge_index is not None:
            # This is a merged parameter that needs to be split
            if training_key not in merge_groups:
                merge_groups[training_key] = []
            merge_groups[training_key].append(
                (diffusers_key, merge_index, num_params_to_merge))

    # Second pass: handle merged parameters by splitting them
    used_keys = set()
    for training_key, splits in merge_groups.items():
        if training_key in state_dict:
            v = state_dict[training_key]
            # Sort by merge_index to ensure correct order
            splits.sort(key=lambda x: x[1])
            total = splits[0][2]
            split_size = v.shape[0] // total
            split_tensors = torch.split(v, split_size, dim=0)

            for diffusers_key, split_index, _ in splits:
                new_state_dict[diffusers_key] = split_tensors[split_index]
            used_keys.add(training_key)

    # Third pass: handle regular parameters (direct mappings)
    for training_key, v in state_dict.items():
        if training_key in used_keys:
            continue

        if training_key in reverse_param_names_mapping:
            diffusers_key, merge_index, _ = reverse_param_names_mapping[
                training_key]
            if merge_index is None:
                # Direct mapping
                new_state_dict[diffusers_key] = v
        else:
            # No mapping found, keep as is
            new_state_dict[training_key] = v

    return new_state_dict


def pred_noise_to_pred_video(pred_noise: torch.Tensor,
                             noise_input_latent: torch.Tensor,
                             timestep: torch.Tensor,
                             scheduler: Any) -> torch.Tensor:
    """
    Convert predicted noise to clean latent.
    """
    timestep = timestep.expand(noise_input_latent.shape[0])
    dtype = pred_noise.dtype
    device = pred_noise.device
    pred_noise = pred_noise.float().to(device)
    noise_input_latent = noise_input_latent.float().to(device)
    # sigmas = scheduler.sigmas.float().to(device)
    # timesteps = scheduler.timesteps.float().to(device)
    # timestep_id = torch.argmin(
    #     (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    # sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    sigma_t = timestep / 1000.0
    sigma_t = sigma_t.reshape(-1, 1, 1, 1)
    # if get_sp_parallel_rank() == 0:
    #     print('sigma:', sigma_t)
    pred_video = noise_input_latent.to(torch.float32) - sigma_t.to(torch.float32) * pred_noise.to(torch.float32)
    return pred_video.to(dtype)


def shift_timestep(timestep: torch.Tensor, shift: float,
                   num_train_timestep: float) -> torch.Tensor:
    if shift == 1:
        return timestep
    t = timestep / num_train_timestep
    denominator = 1 + (shift - 1) * t
    return num_train_timestep * (shift * t / denominator)


# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from diffusers/src/diffusers/optimization.py
"""PyTorch optimization for diffusion models."""


class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    PIECEWISE_CONSTANT = "piecewise_constant"


def get_constant_schedule(optimizer: Optimizer,
                          last_epoch: int = -1) -> LambdaLR:
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer: Optimizer,
                                      num_warmup_steps: int,
                                      last_epoch: int = -1) -> LambdaLR:
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_piecewise_constant_schedule(optimizer: Optimizer,
                                    step_rules: str,
                                    last_epoch: int = -1) -> LambdaLR:
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        step_rules (`string`):
            The rules for the learning rate. ex: rule_steps="1:10,0.1:20,0.01:30,0.005" it means that the learning rate
            if multiple 1 for the first 10 steps, multiple 0.1 for the next 20 steps, multiple 0.01 for the next 30
            steps and multiple 0.005 for the other steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    rules_dict = {}
    rule_list = step_rules.split(",")
    for rule_str in rule_list[:-1]:
        value_str, steps_str = rule_str.split(":")
        steps = int(steps_str)
        value = float(value_str)
        rules_dict[steps] = value
    last_lr_multiple = float(rule_list[-1])

    def create_rules_function(
            rules_dict: dict,
            last_lr_multiple: float) -> Callable[[int], float]:

        def rule_func(steps: int) -> float:
            for step_threshold, lr_multiple in sorted(rules_dict.items()):
                if steps < step_threshold:
                    return lr_multiple
            return last_lr_multiple

        return rule_func

    rules_func = create_rules_function(rules_dict, last_lr_multiple)

    return LambdaLR(optimizer, rules_func, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer: Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    last_epoch: int = -1) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer: Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_min_lr(optimizer: Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    min_lr_ratio: float = 0.1,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to a minimum lr (min_lr_ratio * initial_lr), after a warmup period during which 
    it increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        min_lr_ratio (`float`, *optional*, defaults to 0.1):
            The ratio of minimum learning rate to initial learning rate.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        # Cosine decay from 1.0 to min_lr_ratio over num_cycles periods
        # Use the same formula as standard cosine but ensure minimum is min_lr_ratio instead of 0
        cosine_value = 0.5 * (
            1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        # Ensure the value doesn't go below min_lr_ratio
        return max(min_lr_ratio, cosine_value)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int = 1,
        last_epoch: int = -1) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi *
                                       ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(
            f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Type alias for scheduler functions
SchedulerFunction = Callable[..., LambdaLR]

TYPE_TO_SCHEDULER_FUNCTION: dict[SchedulerType, SchedulerFunction] = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS:
    get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.COSINE_WITH_MIN_LR: get_cosine_schedule_with_min_lr,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.PIECEWISE_CONSTANT: get_piecewise_constant_schedule,
}


def get_scheduler(
    name: str | SchedulerType,
    optimizer: Optimizer,
    step_rules: str | None = None,
    num_warmup_steps: int | None = None,
    num_training_steps: int | None = None,
    num_cycles: int = 1,
    power: float = 1.0,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        step_rules (`str`, *optional*):
            A string representing the step rules to use. This is only used by the `PIECEWISE_CONSTANT` scheduler.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles (`int`, *optional*):
            The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler.
        power (`float`, *optional*, defaults to 1.0):
            Power factor. See `POLYNOMIAL` scheduler
        min_lr_ratio (`float`, *optional*, defaults to 0.1):
            The ratio of minimum learning rate to initial learning rate. Used in `COSINE_WITH_MIN_LR` scheduler.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, last_epoch=last_epoch)

    if name == SchedulerType.PIECEWISE_CONSTANT:
        return schedule_func(optimizer,
                             step_rules=step_rules,
                             last_epoch=last_epoch)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(
            f"{name} requires `num_warmup_steps`, please provide that argument."
        )

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer,
                             num_warmup_steps=num_warmup_steps,
                             last_epoch=last_epoch)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(
            f"{name} requires `num_training_steps`, please provide that argument."
        )

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
            last_epoch=last_epoch,
        )

    if name == SchedulerType.COSINE_WITH_MIN_LR:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )

    return schedule_func(optimizer,
                         num_warmup_steps=num_warmup_steps,
                         num_training_steps=num_training_steps,
                         last_epoch=last_epoch)
