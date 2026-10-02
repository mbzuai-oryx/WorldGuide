import json
import numpy as np
import imageio
import os

import torch
import torch.distributed as dist


def export_to_video(video_frames, output_video_path, fps=12):
    # Export a sequence of video frames to an MP4 file with specified frame rate
    # Ensure all frames are NumPy arrays and determine video dimensions from the first frame
    assert all(
        isinstance(frame, np.ndarray) for frame in video_frames
    ), "All video frames must be NumPy arrays."
    # Ensure output_video_path is ending with .mp4
    if not output_video_path.endswith(".mp4"):
        output_video_path += ".mp4"
    # Create a video file at the specified path and write frames to it
    with imageio.get_writer(output_video_path, fps=fps, format="mp4") as writer:
        for frame in video_frames:
            writer.append_data((frame * 255).astype(np.uint8))


def save_generation(video_frames, configs, base_path, file_name=None):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    p_config = configs["pipe_configs"]
    frames, steps, fps = p_config["num_frames"], p_config["steps"], p_config["fps"]
    if not file_name:
        index = [int(each.split("_")[0]) for each in os.listdir(base_path)]
        max_idex = max(index) if index else 0
        idx_str = str(max_idex + 1).zfill(6)

        key_info = "_".join([str(frames), str(steps), str(fps)])
        file_name = f"{idx_str}_{key_info}"

    with open(f"{base_path}/{file_name}.json", "w") as f:
        json.dump(configs, f, indent=4)

    export_to_video(
        video_frames, f"{base_path}/{file_name}.mp4", fps=p_config["export_fps"]
    )

    return file_name


class GlobalState:
    # Shared state container for passing configuration across distributed components
    def __init__(self, state={}) -> None:
        self.init_state(state)

    def init_state(self, state={}):
        self.state = state

    def set(self, key, value):
        self.state[key] = value

    def get(self, key, default=None):
        return self.state.get(key, default)


class DistController(object):
    # Controller for distributed training/inference across multiple GPUs
    def __init__(self, rank, local_rank, world_size, config=None) -> None:
        super().__init__()
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.config = config
        # Designate rank 0 as the master process for logging and coordination
        self.is_master = rank == 0
        # print("DistController is master: ", self.is_master)
        # self.init_dist()
        self.init_group()
        # self.device = torch.device(f"cuda:{config['devices'][dist.get_rank()]}")
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

    def init_dist(self):
        # print(f"Rank {self.rank} is running.")
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self.config.get("master_port") or "29500")
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

    def init_group(self):
        # Create communication groups for adjacent GPU pairs to enable ring communication
        self.adj_groups = [
            dist.new_group([i, i + 1]) for i in range(self.world_size - 1)
        ]


def distribute_data_to_gpus(data, dim, rank, local_rank, num_gpus, dtype):
    # Partition data along a specified dimension and distribute chunks to multiple GPUs
    dim_size = data.shape[dim]
    assert (
        dim_size % num_gpus == 0
    ), f"Dim {dim_size} of data is not a multiply of {num_gpus}"

    chunk_size = dim_size // num_gpus

    slices = [slice(None)] * len(data.shape)
    slices[dim] = slice(rank * chunk_size, (rank + 1) * chunk_size)

    local_data = data[slices].to(device=f"cuda:{local_rank}", dtype=dtype)
    # print(f"Rank {rank} local_data: {local_data.shape}")
    return local_data


def gather_results(local_result, dim, world_size):
    dist.barrier()

    gathered_results = [torch.zeros_like(local_result) for _ in range(world_size)]
    dist.all_gather(gathered_results, local_result)

    gathered_result = torch.cat(gathered_results, dim=dim)
    return gathered_result
