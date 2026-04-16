from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent.parent


def _is_verbose() -> bool:
    return os.environ.get("VERBOSE_CUDA_BUILD", "0") == "1" or os.environ.get("DIFF_GS_VERBOSE_BUILD", "0") == "1"


@lru_cache(maxsize=1)
def _load_extension():
    return load(
        name="diff_gaussian_rasterization_backend",
        sources=[
            str(_ROOT / "cuda_rasterizer" / "rasterizer_impl.cu"),
            str(_ROOT / "cuda_rasterizer" / "forward.cu"),
            str(_ROOT / "cuda_rasterizer" / "backward.cu"),
            str(_ROOT / "rasterize_points.cu"),
            str(_ROOT / "ext.cpp"),
        ],
        extra_cflags=["-g"],
        extra_cuda_cflags=["-g", "-G", "-I" + str(_ROOT / "third_party" / "glm")],
        verbose=_is_verbose(),
    )


def __getattr__(name: str):
    return getattr(_load_extension(), name)


def __dir__():
    return sorted(set(globals()) | set(dir(_load_extension())))
