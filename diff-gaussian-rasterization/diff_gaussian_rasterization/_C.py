from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent.parent


def _is_verbose() -> bool:
    return os.environ.get("VERBOSE_CUDA_BUILD", "0") == "1" or os.environ.get("DIFF_GS_VERBOSE_BUILD", "0") == "1"


def _is_debug_build() -> bool:
    return os.environ.get("DIFF_GS_DEBUG_BUILD", "0") == "1"


def _extra_cflags() -> list[str]:
    return ["-g"] if _is_debug_build() else ["-O3"]


def _extra_cuda_cflags() -> list[str]:
    include_glm = "-I" + str(_ROOT / "third_party" / "glm")
    if _is_debug_build():
        return ["-g", "-G", include_glm]
    return ["-O3", "--use_fast_math", include_glm]


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
        extra_cflags=_extra_cflags(),
        extra_cuda_cflags=_extra_cuda_cflags(),
        verbose=_is_verbose(),
    )


def __getattr__(name: str):
    return getattr(_load_extension(), name)


def __dir__():
    return sorted(set(globals()) | set(dir(_load_extension())))
