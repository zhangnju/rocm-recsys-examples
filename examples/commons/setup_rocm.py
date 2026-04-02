"""
ROCm/HIP build script for examples/commons CUDA extensions.

Usage (from the examples/commons/ directory):
    python setup_rocm.py build_ext --inplace

Only hstu_cuda_ops (jagged tensor ops used by training) is built.
paged_kvcache_ops (inference only, depends on NVIDIA nvcomp) is skipped.

Supported GPUs: AMD MI355X (gfx950), AMD MI300X/MI300A (gfx942), and other
CDNA AMD GPUs.  The target architecture is auto-detected via
rocm_agent_enumerator; override by setting HIP_ARCHS=gfx942,gfx950 etc.

ROCm: 7.x
"""
import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ---------------------------------------------------------------------------
# Compiler flags for HIP/HIPCC via PyTorch's CUDAExtension (which calls hipcc
# on ROCm builds).  `extra_compile_args["nvcc"]` is forwarded to hipcc.
# ---------------------------------------------------------------------------

# Detect target GPU architectures from the system
def get_hip_arch():
    try:
        out = subprocess.check_output(
            ["/opt/rocm/bin/rocm_agent_enumerator"], stderr=subprocess.DEVNULL
        ).decode().strip().splitlines()
        # Filter out 'gfx000' (CPU placeholder)
        archs = [a.strip() for a in out if a.strip() and a.strip() != "gfx000"]
        return archs or ["gfx942", "gfx950"]
    except Exception:
        return ["gfx942", "gfx950"]  # MI300X / MI355X fallback


HIP_ARCHS = os.environ.get("HIP_ARCHS", ",".join(get_hip_arch())).split(",")
print(f"[setup_rocm] Building for HIP archs: {HIP_ARCHS}")

# One --offload-arch per detected GPU
offload_arch_flags = []
for arch in HIP_ARCHS:
    offload_arch_flags.append(f"--offload-arch={arch}")

hipcc_flags = [
    "-O3",
    "-std=c++17",
    # Enable half / bfloat16 operators (equivalent of -U__CUDA_NO_HALF_OPERATORS__)
    "-D__HIP_NO_HALF_OPERATORS__=0",
    "-D__HIP_NO_HALF_CONVERSIONS__=0",
    "-D__HIP_NO_BFLOAT16_OPERATORS__=0",
    # Relaxed constexpr and lambdas are on by default in clang
    "-ffast-math",
    # Silence some clang-specific warnings
    "-Wno-unused-result",
    "-Wno-deprecated-declarations",
] + offload_arch_flags

cxx_flags = [
    "-O3",
    "-std=c++17",
    "-fvisibility=hidden",
]

setup(
    name="hstu_cuda_ops",
    description="HSTU HIP ops (ROCm build)",
    ext_modules=[
        CUDAExtension(
            name="hstu_cuda_ops",
            sources=[
                "ops/cuda_ops/csrc/jagged_tensor_op_cuda.cpp",
                "ops/cuda_ops/csrc/jagged_tensor_op_kernel.cu",
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": hipcc_flags,
            },
            # Make the local csrc/ directory available so hip_compat.h is found
            include_dirs=[
                os.path.abspath("ops/cuda_ops/csrc"),
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
