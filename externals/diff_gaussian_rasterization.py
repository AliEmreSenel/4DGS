import os


def _is_debug_build() -> bool:
    return os.environ.get("DIFF_GS_DEBUG_BUILD", "0") == "1"


def _extra_cflags() -> list[str]:
    return ["-g"] if _is_debug_build() else ["-O3"]


def _extra_cuda_cflags(parent_dir: str) -> list[str]:
    include_glm = "-I" + os.path.join(parent_dir, "third_party/glm")
    if _is_debug_build():
        return ["-g", "-G", include_glm]
    return ["-O3", "--use_fast_math", include_glm]

try:
    from diff_gaussian_rasterization import _C
except Exception:
    from torch.utils.cpp_extension import load

    parent_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "diff-gaussian-rasterization",
    )
    _C = load(
        name="diff_gaussian_rasterization_backend",
        extra_cflags=_extra_cflags(),
        extra_cuda_cflags=_extra_cuda_cflags(parent_dir),
        sources=[
            os.path.join(parent_dir, "cuda_rasterizer/rasterizer_impl.cu"),
            os.path.join(parent_dir, "cuda_rasterizer/forward.cu"),
            os.path.join(parent_dir, "cuda_rasterizer/backward.cu"),
            os.path.join(parent_dir, "rasterize_points.cu"),
            os.path.join(parent_dir, "ext.cpp"),
        ],
        verbose=os.environ.get("VERBOSE_CUDA_BUILD", "0") == "1"
        or os.environ.get("DIFF_GS_VERBOSE_BUILD", "0") == "1",
    )

__all__ = ["_C"]
