"""Optional Qwen video backend that keeps native video decoding in FFmpeg children.

Qwen's existing frame-count selection, resizing and normalization are retained.
Short nonempty clips repeat sampled frames to satisfy explicit frame requests.
This is a selectable mitigation for native decoder crashes, not proof that a
particular decoder caused a previously recorded SIGSEGV.
"""

import inspect

import imageio.v2 as imageio
import numpy as np
import torch


def read_video_ffmpeg(element):
    from qwen_vl_utils import vision_process as vision

    path = element["video"]
    if path.startswith("file://"):
        path = path[7:]
    reader = imageio.get_reader(path, format="ffmpeg", input_params=["-threads", "1"])
    try:
        total = reader.count_frames()
        fps = float(reader.get_meta_data()["fps"])
        if total < 1 or fps <= 0:
            raise ValueError(f"No usable video frames or FPS in {path}")
        if hasattr(vision, "calculate_video_frame_range"):
            start, end, selected_total = vision.calculate_video_frame_range(element, total, fps)
        else:
            if element.get("video_start") is not None or element.get("video_end") is not None:
                raise ValueError("This qwen-vl-utils version does not support video time ranges.")
            start, end, selected_total = 0, total - 1, total
        # smart_nframes normally rejects an explicit request exceeding the clip
        # length. Some manifest previews contain only one to six frames, while
        # the planner requests eight. Keep Qwen's rounding and validation, but
        # allow uniform sampling with repeated indices for these short clips.
        # This is only a sampling cap: metadata, range bounds and sample FPS
        # below must continue to describe the actual selected source frames.
        sampling_total = selected_total
        if "nframes" in element:
            requested = vision.round_by_factor(element["nframes"], vision.FRAME_FACTOR)
            sampling_total = max(selected_total, requested)
        elif selected_total < vision.FRAME_FACTOR:
            sampling_total = vision.FRAME_FACTOR
        count = vision.smart_nframes(element, total_frames=sampling_total, video_fps=fps)
        indices = torch.linspace(start, end, count).round().long().tolist()
        frames = np.stack([reader.get_data(index) for index in indices])
    finally:
        reader.close()
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    sample_fps = count / selected_total * fps
    metadata = {"fps": fps, "frames_indices": indices, "total_num_frames": selected_total,
                "video_backend": "ffmpeg_subprocess"}
    # qwen-vl-utils added decoder metadata in newer releases.
    if "return_video_metadata" in inspect.signature(vision.fetch_video).parameters:
        return video, metadata, sample_fps
    return video, sample_fps


def configure_planner_video_backend(backend: str) -> None:
    if backend == "qwen":
        return
    if backend != "ffmpeg":
        raise ValueError(f"Unknown planner video backend: {backend}")
    from qwen_vl_utils import vision_process as vision

    # Each worker owns its Qwen backend registry; no other worker is modified.
    vision.VIDEO_READER_BACKENDS["ffmpeg_subprocess"] = read_video_ffmpeg
    # fetch_video hardcodes torchvision as its fallback on read errors. Keep
    # that fallback isolated too, instead of re-entering an in-process codec.
    vision.VIDEO_READER_BACKENDS["torchvision"] = read_video_ffmpeg
    vision.FORCE_QWENVL_VIDEO_READER = "ffmpeg_subprocess"
    vision.get_video_reader_backend.cache_clear()
