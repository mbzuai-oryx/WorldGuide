# CUDA-based device communicator using PyNccl for efficient GPU communication
import torch
from torch.distributed import ProcessGroup

from distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


class CudaCommunicator(DeviceCommunicatorBase):

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)

        from distributed.device_communicators.pynccl import PyNcclCommunicator

        self.pynccl_comm: PyNcclCommunicator | None = None
        # Initialize NCCL communicator for multi-GPU communication
        if self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

    def all_reduce(self, input_, op: torch.distributed.ReduceOp | None = None):
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        # Use PyNccl for optimized GPU all-reduce
        out = pynccl_comm.all_reduce(input_, op=op)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            # Fallback to PyTorch distributed for compatibility
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group, op=op)
        return out

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            # Default to next rank in ring topology
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            # Use PyNccl send for better GPU performance
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            # Default to previous rank in ring topology
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            # Use PyNccl recv for better GPU performance
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self) -> None:
        # Clean up PyNccl communicator resources
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
