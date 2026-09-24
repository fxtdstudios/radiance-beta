"""Frame chunking: process a clip a few frames at a time (3.5.0).

Several VFX nodes built every intermediate for the whole clip at once, so their
peak memory was a multiple of the clip: 240 frames of 4K is 24 GB as one float32
IMAGE, and a node holding five temporaries needs five times that. Working in
chunks bounds the peak by the chunk instead of the clip, and lets the work run
on the GPU (ComfyUI hands IMAGE tensors over on the CPU).
"""
from __future__ import annotations

from typing import Iterator, Tuple

import torch

#: Budget for one chunk's working set when nothing better is known.
CPU_BUDGET_BYTES = 1 << 30          # 1 GiB
#: Share of free VRAM one chunk may use.
GPU_FREE_FRACTION = 0.35


def chunk_budget_bytes(device: torch.device) -> int:
    """Bytes one chunk's working set may use on `device`."""
    if device.type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info(device)
            return max(64 << 20, int(free * GPU_FREE_FRACTION))
        except Exception:  # noqa: BLE001 - fall back to the CPU budget
            pass
    return CPU_BUDGET_BYTES


def frames_per_chunk(height: int, width: int, channels: int, working_copies: float,
                     device: torch.device, bytes_per_value: int = 4) -> int:
    """How many frames fit one chunk, given the number of frame-sized buffers the
    work keeps alive at once (input, output and temporaries)."""
    per_frame = max(1, int(height * width * channels * bytes_per_value * max(1.0, working_copies)))
    return max(1, chunk_budget_bytes(device) // per_frame)


def chunks(batch: int, size: int) -> Iterator[Tuple[int, int]]:
    """(start, end) ranges covering `batch` frames, `size` at a time."""
    size = max(1, int(size))
    for start in range(0, batch, size):
        yield start, min(batch, start + size)


def compute_device() -> torch.device:
    """The device heavy per-pixel work should run on: CUDA, then MPS, then CPU.
    ComfyUI passes IMAGE tensors on the CPU, so using `image.device` leaves the
    GPU idle."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
