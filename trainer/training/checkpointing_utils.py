# SPDX-License-Identifier: Apache-2.0
import random
from typing import Any

import numpy as np
import torch
import torch.distributed.checkpoint.stateful
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_optimizer_state_dict,
                                                     set_model_state_dict,
                                                     set_optimizer_state_dict)


def _canonical_parameter_name(name: str) -> str:
    return (
        name.replace("._checkpoint_wrapped_module.", ".")
        .replace("._orig_mod.", ".")
        .replace(".module.", ".")
    )


def _set_nested_value(container: dict[str, Any], path: list[str], value: Any) -> None:
    current = container
    for key in path[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def _convert_legacy_flattened_optimizer_state_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    if "state" in state_dict and "param_groups" in state_dict:
        return state_dict

    if not state_dict:
        return state_dict

    if not any(key.startswith("state.") or key.startswith("param_groups.") for key in state_dict):
        return state_dict

    param_names = [
        _canonical_parameter_name(name)
        for name, _ in model.named_parameters()
    ]
    param_name_set = set(param_names)
    name_by_param_id = {
        id(param): _canonical_parameter_name(name)
        for name, param in model.named_parameters()
    }

    def split_param_key(body: str) -> tuple[str, str]:
        for param_name in sorted(param_name_set, key=len, reverse=True):
            if body == param_name:
                return param_name, ""
            if body.startswith(param_name + "."):
                return param_name, body[len(param_name) + 1:]
        raise KeyError(f"Could not map optimizer checkpoint key '{body}' to a model parameter name.")

    nested_state: dict[str, Any] = {}
    nested_group_values: dict[str, dict[str, Any]] = {}

    for key, value in state_dict.items():
        if key.startswith("state."):
            param_name, remainder = split_param_key(key[len("state."):])
            if not remainder:
                raise KeyError(f"Malformed legacy optimizer state key: {key}")
            state_entry = nested_state.setdefault(param_name, {})
            _set_nested_value(state_entry, remainder.split("."), value)
        elif key.startswith("param_groups."):
            param_name, remainder = split_param_key(key[len("param_groups."):])
            if not remainder:
                raise KeyError(f"Malformed legacy optimizer param_groups key: {key}")
            group_entry = nested_group_values.setdefault(param_name, {})
            _set_nested_value(group_entry, remainder.split("."), value)

    nested_param_groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        group_param_names = [name_by_param_id[id(param)] for param in group["params"]]
        nested_group: dict[str, Any] = {"params": group_param_names}
        for param_name in group_param_names:
            per_param_group_values = nested_group_values.get(param_name, {})
            for attr_key, attr_value in per_param_group_values.items():
                if attr_key in nested_group and nested_group[attr_key] != attr_value:
                    raise ValueError(
                        "Legacy flattened optimizer checkpoint stores inconsistent "
                        f"param-group attribute '{attr_key}' across parameters in the same group."
                    )
                nested_group[attr_key] = attr_value
        nested_param_groups.append(nested_group)

    return {
        "state": nested_state,
        "param_groups": nested_param_groups,
    }


class ModelWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self,
                 model: torch.nn.Module,
                 cpu_offload: bool = False,
                 full_state_dict: bool = False,
                 broadcast_from_rank0: bool = False) -> None:
        self.model = model
        self.cpu_offload = cpu_offload
        self.full_state_dict = full_state_dict
        self.broadcast_from_rank0 = broadcast_from_rank0

    def state_dict(self) -> dict[str, Any]:
        state_dict = get_model_state_dict(
            self.model,
            options=StateDictOptions(
                cpu_offload=self.cpu_offload,
                full_state_dict=self.full_state_dict,
            ),
        )  # type: ignore[no-any-return]
        # filter out non-trainable parameters
        param_requires_grad = set([
            k for k, v in dict(self.model.named_parameters()).items()
            if v.requires_grad
        ])
        state_dict = {
            k: v
            for k, v in state_dict.items() if k in param_requires_grad
        }
        return state_dict  # type: ignore

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(
                cpu_offload=self.cpu_offload,
                strict=False,
                full_state_dict=self.full_state_dict,
                broadcast_from_rank0=self.broadcast_from_rank0,
            ),
        )


class OptimizerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 cpu_offload: bool = False,
                 full_state_dict: bool = False,
                 broadcast_from_rank0: bool = False) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cpu_offload = cpu_offload
        self.full_state_dict = full_state_dict
        self.broadcast_from_rank0 = broadcast_from_rank0

    def state_dict(self) -> dict[str, Any]:
        return get_optimizer_state_dict(  # type: ignore[no-any-return]
            self.model,
            self.optimizer,
            options=StateDictOptions(
                cpu_offload=self.cpu_offload,
                full_state_dict=self.full_state_dict,
                flatten_optimizer_state_dict=False,
            ),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.full_state_dict and hasattr(self.optimizer, "materialize_state_tensors"):
            self.optimizer.materialize_state_tensors(lightweight=True)
        state_dict = _convert_legacy_flattened_optimizer_state_dict(
            self.model,
            self.optimizer,
            state_dict,
        )
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(
                cpu_offload=self.cpu_offload,
                full_state_dict=self.full_state_dict,
                broadcast_from_rank0=self.broadcast_from_rank0,
                flatten_optimizer_state_dict=False,
            ),
        )


class SchedulerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def state_dict(self) -> dict[str, Any]:
        return {"scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict["scheduler"])


class RandomStateWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, noise_generator: torch.Generator | None = None) -> None:
        self.noise_generator = noise_generator

    def state_dict(self) -> dict[str, Any]:
        state = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
            if torch.cuda.device_count() > 1:
                state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        if self.noise_generator is not None:
            state["noise_generator_state"] = self.noise_generator.get_state()

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "torch_rng_state" in state_dict:
            torch.set_rng_state(state_dict["torch_rng_state"])

        if "numpy_rng_state" in state_dict:
            np.random.set_state(state_dict["numpy_rng_state"])

        if "python_rng_state" in state_dict:
            random.setstate(state_dict["python_rng_state"])

        # Restore CUDA random state
        if torch.cuda.is_available():
            if "cuda_rng_state" in state_dict:
                torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
            if "cuda_rng_state_all" in state_dict:
                torch.cuda.set_rng_state_all(state_dict["cuda_rng_state_all"])

        # Restore noise generator state
        if "noise_generator_state" in state_dict and self.noise_generator is not None:
            self.noise_generator.set_state(state_dict["noise_generator_state"])
