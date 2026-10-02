import torch
import torch.distributed as dist
import math

import time


class ModulePlugin:
    def __init__(self, module, module_id, global_state=None):
        self.module = module
        self.module_id = module_id
        self.global_state = global_state
        self.enable = True
        self.implement_forward()

    @property
    def is_log_node(self):
        return (
            self.global_state.get("dist_controller").rank == 0
            and self.module_id[1] == 0
        )

    @property
    def t(self):
        return self.global_state.get("timestep")

    @property
    def p(self):
        return self.t / 1000

    def implement_forward(self):
        module = self.module
        if not hasattr(module, "old_forward"):
            module.old_forward = module.forward
        self.new_forward = self.get_new_forward()

        def forward(*args, **kwargs):
            self.update_config()  # update config
            return (
                self.new_forward(*args, **kwargs)
                if self.enable
                else self.old_forward(*args, **kwargs)
            )

        module.forward = forward

    def set_enable(self, enable=True):
        self.enable = enable

    def get_new_forward(self):
        raise NotImplementedError

    def update_config(self, config: dict = None):
        if config is None:
            config = self.global_state.get("plugin_configs", {}).get(
                self.module_id[0], {}
            )
        for key, value in config.items():
            setattr(self, key, value)

    def _calculate_padding(self, kernel_size, chunk_dim):
        """Calculate padding parameters for convolution layers.

        Args:
            kernel_size: kernel size tuple or int
            chunk_dim: dimension to chunk/pad along

        Returns:
            tuple: (padding_left, padding_right, padding_flag)
        """
        if isinstance(kernel_size, int):
            kernel_size = tuple([kernel_size] * (chunk_dim - 1))

        # For 3D conv, use width dimension (index 2)
        # For 2D conv, calculate based on chunk_dim
        if len(kernel_size) == 3:
            kernel_width = kernel_size[2]
        else:
            pad_kernel_dim = chunk_dim - 3
            kernel_width = kernel_size[pad_kernel_dim]

        d = kernel_width - 1
        padding_left = d // 2
        padding_right = d - padding_left
        padding_flag = padding_left if d > 0 else 0

        return padding_left, padding_right, padding_flag

    def _pad_context_bilateral_impl(
        self, h, padding_left, padding_right, slice_dim, cat_dim
    ):
        """Core implementation for bilateral padding with context from adjacent ranks.

        Args:
            h: input tensor to pad
            padding_left: number of elements to pad on left
            padding_right: number of elements to pad on right
            slice_dim: dimension to slice from (3 for 3D, 3 for 2D height/width)
            cat_dim: dimension to concatenate along

        Returns:
            padded tensor with context from adjacent ranks
        """
        if padding_left == 0:
            return h

        # Extract slices to share with neighbors
        if slice_dim == 3:
            share_to_left = h[:, :, :, :padding_left].contiguous()
            share_to_right = h[:, :, :, -padding_right:].contiguous()
        elif slice_dim == 4:
            share_to_left = h[:, :, :, :, :padding_left].contiguous()
            share_to_right = h[:, :, :, :, -padding_right:].contiguous()
        elif slice_dim == 2:
            share_to_left = h[:, :, :padding_left].contiguous()
            share_to_right = h[:, :, -padding_right:].contiguous()
        else:  # slice_dim == 1
            share_to_left = h[:, :, :, :padding_left].contiguous()
            share_to_right = h[:, :, :, -padding_right:].contiguous()

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        adj_groups = self.global_state.get("dist_controller").adj_groups

        if rank % 2:
            # Odd rank: pad left first, then right
            if rank:
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=adj_groups[rank - 1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(
                    h.device, non_blocking=True
                )

            if rank != world_size - 1:
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=adj_groups[rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(
                    h.device, non_blocking=True
                )
        else:
            # Even rank: pad right first, then left
            if rank != world_size - 1:
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=adj_groups[rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(
                    h.device, non_blocking=True
                )

            if rank:
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=adj_groups[rank - 1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(
                    h.device, non_blocking=True
                )

        h_with_context = torch.cat([left_context, h, right_context], dim=cat_dim)
        return h_with_context

    def _pad_context_bilateral(self, h, padding_left, padding_right, chunk_dim):
        """Pad tensor with context from adjacent ranks using bilateral communication.

        Args:
            h: input tensor to pad
            padding_left: number of elements to pad on left
            padding_right: number of elements to pad on right
            chunk_dim: dimension to pad along

        Returns:
            padded tensor with context from adjacent ranks
        """
        slice_dim = 3 if chunk_dim == 3 else 4
        return self._pad_context_bilateral_impl(
            h, padding_left, padding_right, slice_dim, chunk_dim
        )

    def _pad_context_bilateral_2d(self, h, padding_left, padding_right, chunk_dim):
        """Pad 2D tensor with context from adjacent ranks using bilateral communication.

        Similar to _pad_context_bilateral but pads on chunk_dim-1 dimension for 2D convolutions.

        Args:
            h: input tensor to pad
            padding_left: number of elements to pad on left
            padding_right: number of elements to pad on right
            chunk_dim: base dimension reference (actual padding on chunk_dim-1)

        Returns:
            padded tensor with context from adjacent ranks
        """
        slice_dim = 2 if chunk_dim == 3 else 1
        return self._pad_context_bilateral_impl(
            h, padding_left, padding_right, slice_dim, chunk_dim - 1
        )


class GroupNormPlugin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)

    def get_new_forward(self):
        module = self.module

        def new_forward(x):
            shape = x.shape
            N, C, G = shape[0], shape[1], module.num_groups
            assert C % G == 0

            x = x.reshape(N, G, -1)

            mean = x.mean(-1, keepdim=True).to(torch.float32)
            dist.all_reduce(mean)

            mean = mean / dist.get_world_size()

            var = ((x - mean.to(x.dtype)) ** 2).mean(-1, keepdim=True).to(torch.float32)

            dist.all_reduce(var)
            var = var / dist.get_world_size()

            x = (x - mean.to(x.dtype)) / (var.to(x.dtype) + module.eps).sqrt()
            x = x.view(shape)

            new_shape = [1 for _ in shape]
            new_shape[1] = -1

            return x * module.weight.view(new_shape) + module.bias.view(new_shape)

        return new_forward


class Conv3DSafeNewPligin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)

        self.kernel_size = getattr(module, "kernel_size", (1, 1, 1))

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)

        self.chunk_dim = global_state.get("chunk_dim")
        self.padding_left, self.padding_right, self.padding_flag = (
            self._calculate_padding(self.kernel_size, self.chunk_dim)
        )

    def pad_context(self, h):
        return self._pad_context_bilateral(
            h, self.padding_left, self.padding_right, self.chunk_dim
        )

    def get_new_forward(self):
        module = self.module

        def new_forward(hidden_states, cache_x=None, *args, **kwargs):
            if self.padding_flag == 0:
                # print(f"padding=0, return old_forward")
                return module.old_forward(hidden_states, cache_x, *args, **kwargs)

            hidden_states = self.pad_context(hidden_states)
            if cache_x is not None:
                cache_x = self.pad_context(cache_x)

            result = module.old_forward(hidden_states, cache_x, *args, **kwargs)
            if self.chunk_dim == 3:
                result = result[
                    :,
                    :,
                    :,
                    self.padding_left : (
                        -self.padding_right if self.padding_right > 0 else None
                    ),
                ]
            else:
                result = result[
                    :,
                    :,
                    :,
                    :,
                    self.padding_left : (
                        -self.padding_right if self.padding_right > 0 else None
                    ),
                ]

            return result

        return new_forward


class Conv2DSafeNewPligin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)

        self.kernel_size = getattr(module, "kernel_size", (1, 1))
        self.stride = getattr(module, "stride", (1, 1))

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        self.chunk_dim = global_state.get("chunk_dim")
        pad_kernel_dim = self.chunk_dim - 3
        kernel_height = self.kernel_size[pad_kernel_dim]
        d = kernel_height - 1
        self.padding_left = d // 2
        self.padding_right = d - self.padding_left
        self.padding = self.padding_left if d > 0 else 0

    def pad_context(self, h):
        return self._pad_context_bilateral_2d(
            h, self.padding_left, self.padding_right, self.chunk_dim
        )

    def get_new_forward(self):
        module = self.module

        def new_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            if self.padding == 0:
                return module.old_forward(hidden_states)

            hidden_states = self.pad_context(hidden_states)

            if self.chunk_dim == 3:
                hidden_states = module.old_forward(hidden_states)[
                    :,
                    :,
                    self.padding_left : (
                        -self.padding_right if self.padding_right > 0 else None
                    ),
                ]
            else:
                hidden_states = module.old_forward(hidden_states)[
                    :,
                    :,
                    :,
                    self.padding_left : (
                        -self.padding_right if self.padding_right > 0 else None
                    ),
                ]
            return hidden_states

        return new_forward


class Conv2DSafeNewPliginStride2(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)

        self.kernel_size = getattr(module, "kernel_size", (1, 1))
        self.stride = getattr(module, "stride", (1, 1))

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        self.chunk_dim = global_state.get("chunk_dim")
        assert self.kernel_size[0] == self.kernel_size[1]

        kernel_height = self.kernel_size[0]
        d = kernel_height - 1
        self.padding_left = d // 2
        self.padding_right = d - self.padding_left
        self.padding = self.padding_left if d > 0 else 0

    def pad_context(self, h):
        if self.padding == 0:
            return h

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if self.chunk_dim == 3:
            share_to_left = h[:, :, : self.padding_left].contiguous()
        else:
            share_to_left = h[:, :, :, : self.padding_left].contiguous()

        if rank < world_size - 1:
            right_context = torch.zeros_like(share_to_left)

            dist.recv(right_context, src=rank + 1)
        if rank > 0:
            dist.send(share_to_left, dst=rank - 1)
        # torch.cuda.synchronize()
        if rank < world_size - 1:
            h_with_context = torch.cat([h, right_context], dim=self.chunk_dim - 1)
        else:
            h_with_context = h
        return h_with_context

    def get_new_forward(self):
        module = self.module

        def new_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            # assert self.global_state["chunk_dim"] == 3 # only support chunk height now
            if self.padding == 0:
                return module.old_forward(hidden_states)

            if self.chunk_dim == 3:
                hidden_states = hidden_states[:, :, :-1, :]
            else:
                hidden_states = hidden_states[:, :, :, :-1]

            hidden_states = self.pad_context(hidden_states)

            if self.chunk_dim == 3:
                hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, 1))
            else:
                hidden_states = torch.nn.functional.pad(hidden_states, (0, 1))
            hidden_states = module.old_forward(
                hidden_states
            )  # [:,:,self.padding_left:-self.padding_right if self.padding_right > 0 else None]
            return hidden_states

        return new_forward


class WanAttentionPlugin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.gather_dim = global_state.get("chunk_dim")

        super().__init__(module, module_id, global_state)

    def get_new_forward(self):
        module = self.module
        rank = self.rank
        world_size = self.world_size

        def new_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            gathered_tensors = [
                torch.zeros_like(hidden_states) for _ in range(world_size)
            ]
            dist.all_gather(gathered_tensors, hidden_states)

            combined_tensor = torch.cat(gathered_tensors, dim=self.gather_dim)
            forward_output = module.old_forward(combined_tensor)

            # chunk_sizes = [t.size(self.gather_dim) for t in gathered_tensors]
            # start_idx = sum(chunk_sizes[:rank])
            # end_idx = start_idx + chunk_sizes[rank]
            # local_output = forward_output[:, :, :, start_idx:end_idx].contiguous()
            local_output = forward_output.chunk(chunks=world_size, dim=self.gather_dim)[
                rank
            ]

            return local_output

        return new_forward
