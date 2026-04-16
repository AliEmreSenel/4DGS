from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"


def _is_verbose() -> bool:
    return os.environ.get("VERBOSE_CUDA_BUILD", "0") == "1" or os.environ.get("POINTOPS2_VERBOSE_BUILD", "0") == "1"


def _is_debug_build() -> bool:
    return os.environ.get("POINTOPS2_DEBUG_BUILD", "0") == "1"


def _extra_cflags() -> list[str]:
    return ["-g"] if _is_debug_build() else ["-O3"]


def _extra_cuda_cflags() -> list[str]:
    return ["-g", "-G"] if _is_debug_build() else ["-O3", "--use_fast_math"]


def _sources() -> list[str]:
    cpp_sources = sorted(str(path) for path in _SRC.rglob("*.cpp"))
    cuda_sources = sorted(str(path) for path in _SRC.rglob("*.cu"))
    return cpp_sources + cuda_sources


@lru_cache(maxsize=1)
def _load_extension():
    return load(
        name="pointops2_cuda_backend",
        sources=_sources(),
        extra_cflags=_extra_cflags(),
        extra_cuda_cflags=_extra_cuda_cflags(),
        verbose=_is_verbose(),
    )


def __getattr__(name: str):
    return getattr(_load_extension(), name)


def __dir__():
    return sorted(set(globals()) | set(dir(_load_extension())))
