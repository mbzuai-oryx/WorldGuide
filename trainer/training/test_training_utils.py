# SPDX-License-Identifier: Apache-2.0
import random
from pathlib import Path

import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchdata.stateful_dataloader import StatefulDataLoader

from trainer.training.muon import get_muon_optimizer
from trainer.training.checkpointing_utils import (
    _canonical_parameter_name,
    _convert_legacy_flattened_optimizer_state_dict,
)
from trainer.training.training_utils import (
    TRAINING_STATE_FILENAME,
    load_checkpoint_metadata,
    load_checkpoint,
    prune_old_checkpoints,
    resolve_training_schedule,
    save_checkpoint,
)


class DummyStatefulDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.values = list(range(6))
        self.rng = random.Random(1234)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return {
            "value": self.values[idx],
            "rng": self.rng.random(),
        }

    def state_dict(self):
        return {"rng_state": self.rng.getstate()}

    def load_state_dict(self, state_dict):
        self.rng.setstate(state_dict["rng_state"])


def test_prune_old_checkpoints_keeps_latest_n(tmp_path: Path):
    for step in (100, 200, 300):
        checkpoint_dir = tmp_path / f"checkpoint-{step}"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "marker.txt").write_text(str(step), encoding="utf-8")

    prune_old_checkpoints(str(tmp_path), rank=0, save_limit=2)

    remaining = sorted(path.name for path in tmp_path.iterdir())
    assert remaining == ["checkpoint-200", "checkpoint-300"]


def test_load_checkpoint_metadata_reads_json(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoint-100"
    checkpoint_dir.mkdir()
    metadata_path = checkpoint_dir / "checkpoint_metadata.json"
    metadata_path.write_text('{"step": 100, "samples_seen_global": 800}', encoding="utf-8")

    metadata = load_checkpoint_metadata(str(checkpoint_dir))

    assert metadata == {"step": 100, "samples_seen_global": 800}


def test_save_and_load_checkpoint_round_trip_restores_training_state(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    dataset = DummyStatefulDataset()
    dataloader = StatefulDataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    iterator = iter(dataloader)
    next(iterator)
    next(iterator)

    input_tensor = torch.ones(1, 2)
    loss = model(input_tensor).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    noise_generator = torch.Generator(device="cpu").manual_seed(77)
    validation_generator = torch.Generator(device="cpu").manual_seed(88)
    saved_parameters = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    save_checkpoint(
        model,
        rank=0,
        output_dir=str(tmp_path),
        step=2,
        optimizer=optimizer,
        dataloader=dataloader,
        scheduler=scheduler,
        noise_generator=noise_generator,
        validation_random_generator=validation_generator,
        current_epoch=0,
        save_training_state=True,
        save_consolidated_checkpoint=False,
        metadata={
            "train_batch_size": 1,
            "train_sp_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "sp_size": 4,
            "total_batch_size": 2,
            "samples_seen_global": 4,
        },
    )

    checkpoint_dir = tmp_path / "checkpoint-2"
    assert (checkpoint_dir / TRAINING_STATE_FILENAME).exists()
    assert not (checkpoint_dir / "training_state_rank0.pt").exists()
    expected_next_batch = next(iterator)
    saved_training_state = torch.load(
        checkpoint_dir / TRAINING_STATE_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    assert {
        "current_epoch",
        "checkpoint_metadata",
        "dataloader_state",
        "dataset_state",
        "gradient_accumulation_steps",
        "model",
        "samples_seen_global",
        "sp_size",
        "step",
        "total_batch_size",
        "train_batch_size",
        "train_sp_batch_size",
        "world_size",
    }.issubset(saved_training_state.keys())

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = Adam(restored_model.parameters(), lr=1e-3)
    restored_scheduler = LambdaLR(restored_optimizer, lr_lambda=lambda _: 1.0)
    restored_dataset = DummyStatefulDataset()
    restored_dataloader = StatefulDataLoader(
        restored_dataset, batch_size=1, num_workers=0, shuffle=False
    )
    restored_noise_generator = torch.Generator(device="cpu").manual_seed(999)
    restored_validation_generator = torch.Generator(device="cpu").manual_seed(1000)

    resume_state = load_checkpoint(
        restored_model,
        rank=0,
        checkpoint_path=str(checkpoint_dir),
        optimizer=restored_optimizer,
        dataloader=restored_dataloader,
        scheduler=restored_scheduler,
        noise_generator=restored_noise_generator,
        validation_random_generator=restored_validation_generator,
    )

    assert resume_state == {"step": 2, "current_epoch": 0}
    for name, param in restored_model.named_parameters():
        assert torch.equal(param, saved_parameters[name])
    assert restored_scheduler.state_dict() != scheduler.state_dict()
    assert not torch.equal(
        restored_noise_generator.get_state(),
        noise_generator.get_state(),
    )
    assert not torch.equal(
        restored_validation_generator.get_state(),
        validation_generator.get_state(),
    )
    restored_iterator = iter(restored_dataloader)
    restored_next_batch = next(restored_iterator)
    assert restored_next_batch["value"].item() == expected_next_batch["value"].item()
    assert torch.allclose(restored_next_batch["rng"], expected_next_batch["rng"])


def test_load_checkpoint_with_reset_dataset_restarts_from_beginning(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    dataset = DummyStatefulDataset()
    dataloader = StatefulDataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    iterator = iter(dataloader)
    next(iterator)
    next(iterator)

    input_tensor = torch.ones(1, 2)
    loss = model(input_tensor).sum()
    loss.backward()

    save_checkpoint(
        model,
        rank=0,
        output_dir=str(tmp_path),
        step=2,
        dataloader=dataloader,
        current_epoch=1,
        save_training_state=True,
        save_consolidated_checkpoint=False,
        metadata={
            "train_batch_size": 1,
            "train_sp_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "sp_size": 4,
            "total_batch_size": 2,
            "samples_seen_global": 4,
        },
    )

    checkpoint_dir = tmp_path / "checkpoint-2"
    restored_model = torch.nn.Linear(2, 1)
    restored_dataset = DummyStatefulDataset()
    restored_dataloader = StatefulDataLoader(
        restored_dataset, batch_size=1, num_workers=0, shuffle=False
    )

    resume_state = load_checkpoint(
        restored_model,
        rank=0,
        checkpoint_path=str(checkpoint_dir),
        dataloader=restored_dataloader,
        reset_dataset=True,
    )

    assert resume_state == {"step": 2, "current_epoch": 1}
    restarted_batch = next(iter(restored_dataloader))
    assert restarted_batch["value"].item() == 0


def test_resolve_training_schedule_prefers_max_train_steps():
    max_train_steps, num_train_epochs = resolve_training_schedule(
        max_train_steps=250,
        num_train_epochs=2,
        num_update_steps_per_epoch=100,
    )

    assert max_train_steps == 250
    assert num_train_epochs == 3


def test_resolve_training_schedule_uses_num_train_epochs():
    max_train_steps, num_train_epochs = resolve_training_schedule(
        max_train_steps=0,
        num_train_epochs=2,
        num_update_steps_per_epoch=100,
    )

    assert max_train_steps == 200
    assert num_train_epochs == 2


def test_resolve_training_schedule_handles_none_values():
    max_train_steps, num_train_epochs = resolve_training_schedule(
        max_train_steps=None,
        num_train_epochs=2,
        num_update_steps_per_epoch=100,
    )

    assert max_train_steps == 200
    assert num_train_epochs == 2


def test_resolve_training_schedule_requires_positive_limit():
    with pytest.raises(ValueError, match="Either max_train_steps or num_train_epochs"):
        resolve_training_schedule(
            max_train_steps=0,
            num_train_epochs=0,
            num_update_steps_per_epoch=100,
        )


def test_muon_materialize_state_tensors_initializes_tensor_state():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.LayerNorm(4),
    )
    optimizer = get_muon_optimizer(model, lr=1e-3)

    optimizer.materialize_state_tensors()

    saw_muon_buffer = False
    saw_adamw_state = False
    for param in model.parameters():
        state = optimizer.state[param]
        if state["use_muon"]:
            assert "momentum_buffer" in state
            assert isinstance(state["momentum_buffer"], torch.Tensor)
            saw_muon_buffer = True
        else:
            assert state["step"] == 0
            assert isinstance(state["moment1"], torch.Tensor)
            assert isinstance(state["moment2"], torch.Tensor)
            saw_adamw_state = True

    assert saw_muon_buffer
    assert saw_adamw_state


def test_muon_materialize_state_tensors_lightweight_only_seeds_minimal_tensor_state():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.LayerNorm(4),
    )
    optimizer = get_muon_optimizer(model, lr=1e-3)

    optimizer.materialize_state_tensors(lightweight=True)

    tensor_state_count = 0
    for param in model.parameters():
        state = optimizer.state[param]
        for value in state.values():
            if isinstance(value, torch.Tensor):
                tensor_state_count += 1

    assert tensor_state_count >= 1
    assert tensor_state_count <= 2


def test_convert_legacy_flattened_optimizer_state_dict_recovers_nested_format():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.LayerNorm(4),
    )
    optimizer = get_muon_optimizer(model, lr=1e-3)
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param)
    optimizer.step()

    nested_state = optimizer.state_dict()
    legacy_flattened = {
        "state.0.weight.use_muon": True,
        "state.0.weight.momentum_buffer": nested_state["state"][0]["momentum_buffer"],
        "state.0.bias.use_muon": False,
        "state.0.bias.step": nested_state["state"][1]["step"],
        "state.0.bias.moment1": nested_state["state"][1]["moment1"],
        "state.0.bias.moment2": nested_state["state"][1]["moment2"],
        "state.1.weight.use_muon": False,
        "state.1.weight.step": nested_state["state"][2]["step"],
        "state.1.weight.moment1": nested_state["state"][2]["moment1"],
        "state.1.weight.moment2": nested_state["state"][2]["moment2"],
        "state.1.bias.use_muon": False,
        "state.1.bias.step": nested_state["state"][3]["step"],
        "state.1.bias.moment1": nested_state["state"][3]["moment1"],
        "state.1.bias.moment2": nested_state["state"][3]["moment2"],
        "param_groups.0.weight.lr": nested_state["param_groups"][0]["lr"],
        "param_groups.0.weight.wd": nested_state["param_groups"][0]["wd"],
        "param_groups.0.weight.momentum": nested_state["param_groups"][0]["momentum"],
        "param_groups.0.weight.nesterov": nested_state["param_groups"][0]["nesterov"],
        "param_groups.0.weight.ns_steps": nested_state["param_groups"][0]["ns_steps"],
        "param_groups.0.weight.adamw_betas": nested_state["param_groups"][0]["adamw_betas"],
        "param_groups.0.weight.adamw_eps": nested_state["param_groups"][0]["adamw_eps"],
        "param_groups.0.bias.lr": nested_state["param_groups"][0]["lr"],
        "param_groups.0.bias.wd": nested_state["param_groups"][0]["wd"],
        "param_groups.0.bias.momentum": nested_state["param_groups"][0]["momentum"],
        "param_groups.0.bias.nesterov": nested_state["param_groups"][0]["nesterov"],
        "param_groups.0.bias.ns_steps": nested_state["param_groups"][0]["ns_steps"],
        "param_groups.0.bias.adamw_betas": nested_state["param_groups"][0]["adamw_betas"],
        "param_groups.0.bias.adamw_eps": nested_state["param_groups"][0]["adamw_eps"],
        "param_groups.1.weight.lr": nested_state["param_groups"][0]["lr"],
        "param_groups.1.weight.wd": nested_state["param_groups"][0]["wd"],
        "param_groups.1.weight.momentum": nested_state["param_groups"][0]["momentum"],
        "param_groups.1.weight.nesterov": nested_state["param_groups"][0]["nesterov"],
        "param_groups.1.weight.ns_steps": nested_state["param_groups"][0]["ns_steps"],
        "param_groups.1.weight.adamw_betas": nested_state["param_groups"][0]["adamw_betas"],
        "param_groups.1.weight.adamw_eps": nested_state["param_groups"][0]["adamw_eps"],
        "param_groups.1.bias.lr": nested_state["param_groups"][0]["lr"],
        "param_groups.1.bias.wd": nested_state["param_groups"][0]["wd"],
        "param_groups.1.bias.momentum": nested_state["param_groups"][0]["momentum"],
        "param_groups.1.bias.nesterov": nested_state["param_groups"][0]["nesterov"],
        "param_groups.1.bias.ns_steps": nested_state["param_groups"][0]["ns_steps"],
        "param_groups.1.bias.adamw_betas": nested_state["param_groups"][0]["adamw_betas"],
        "param_groups.1.bias.adamw_eps": nested_state["param_groups"][0]["adamw_eps"],
    }

    converted = _convert_legacy_flattened_optimizer_state_dict(
        model,
        optimizer,
        legacy_flattened,
    )

    assert set(converted.keys()) == {"state", "param_groups"}
    assert converted["param_groups"][0]["params"] == ["0.weight", "0.bias", "1.weight", "1.bias"]
    assert converted["param_groups"][0]["lr"] == nested_state["param_groups"][0]["lr"]
    assert torch.equal(
        converted["state"]["0.weight"]["momentum_buffer"],
        nested_state["state"][0]["momentum_buffer"],
    )


def test_canonical_parameter_name_strips_checkpoint_wrapper_prefixes():
    assert (
        _canonical_parameter_name(
            "double_blocks.0._checkpoint_wrapped_module.img_mod.linear.weight"
        )
        == "double_blocks.0.img_mod.linear.weight"
    )
    assert (
        _canonical_parameter_name(
            "foo._orig_mod.bar.module.weight"
        )
        == "foo.bar.weight"
    )
