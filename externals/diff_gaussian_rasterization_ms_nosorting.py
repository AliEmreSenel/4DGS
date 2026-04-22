import os


def _is_debug_build() -> bool:
    return os.environ.get("DIFF_GS_DEBUG_BUILD", "0") == "1"


def _extra_cflags() -> list[str]:
    return ["-g"] if _is_debug_build() else ["-O3"]


def _find_source_root() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, "diff-gaussian-rasterization-ms-nosorting")
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        "MobileGS no-sorting CUDA sources were not found. "
        "Expected diff-gaussian-rasterization-ms-nosorting in repository root."
    )


def _extra_cuda_cflags(source_root: str) -> list[str]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    include_glm = "-I" + os.path.join(repo_root, "diff-gaussian-rasterization", "third_party", "glm")
    if _is_debug_build():
        return ["-g", "-G", include_glm]
    return ["-O3", "--use_fast_math", include_glm]


try:
    from diff_gaussian_rasterization_ms_nosorting import _C
except Exception:
    from torch.utils.cpp_extension import load

    source_root = _find_source_root()
    _C = load(
        name="diff_gaussian_rasterization_ms_nosorting_backend",
        extra_cflags=_extra_cflags(),
        extra_cuda_cflags=_extra_cuda_cflags(source_root),
        sources=[
            os.path.join(source_root, "cuda_rasterizer", "rasterizer_impl.cu"),
            os.path.join(source_root, "cuda_rasterizer", "forward.cu"),
            os.path.join(source_root, "cuda_rasterizer", "backward.cu"),
            os.path.join(source_root, "rasterize_points.cu"),
            os.path.join(source_root, "ext.cpp"),
        ],
        verbose=os.environ.get("VERBOSE_CUDA_BUILD", "0") == "1"
        or os.environ.get("DIFF_GS_VERBOSE_BUILD", "0") == "1",
    )


__all__ = ["_C"]
