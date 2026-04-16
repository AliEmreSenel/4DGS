from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent.parent


def _is_verbose() -> bool:
    return os.environ.get("VERBOSE_CUDA_BUILD", "0") == "1" or os.environ.get("SIMPLE_KNN_VERBOSE_BUILD", "0") == "1"


@lru_cache(maxsize=1)
def _load_extension():
    return load(
        name="simple_knn_backend",
        sources=[
            str(_ROOT / "spatial.cu"),
            str(_ROOT / "simple_knn.cu"),
            str(_ROOT / "ext.cpp"),
        ],
        verbose=_is_verbose(),
    )


def __getattr__(name: str):
    return getattr(_load_extension(), name)


def __dir__():
    return sorted(set(globals()) | set(dir(_load_extension())))


def distCUDA2(*args, **kwargs):
    return _load_extension().distCUDA2(*args, **kwargs)
