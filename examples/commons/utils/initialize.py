# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import os

import torch

try:
    from megatron.core import parallel_state, tensor_parallel
except ImportError:
    print("megatron.core is not installed, training is not supported.")
    parallel_state = None
    tensor_parallel = None


from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Architecture detection utilities
# ---------------------------------------------------------------------------

# Known ROCm architectures that require fbgemm / TBE workarounds.
# gfx950 (MI355X): fbgemm ops SIGSEGV + TBE deadlock.
# Extend this set if new architectures exhibit the same issues.
_ARCHS_NEEDING_FBGEMM_PATCHES: frozenset = frozenset({"gfx950"})
_ARCHS_NEEDING_TBE_BYPASS:     frozenset = frozenset({"gfx950"})


def get_rocm_arch(device: int = 0) -> str:
    """Return the base GCN arch string for *device* (e.g. 'gfx950', 'gfx942').

    Returns an empty string on non-ROCm platforms or when the arch cannot be
    determined.
    """
    if not torch.version.hip:
        return ""
    try:
        props = torch.cuda.get_device_properties(device)
        # gcnArchName looks like 'gfx950:sramecc+:xnack-' – take the base part
        return getattr(props, "gcnArchName", "").split(":")[0].strip()
    except Exception:
        return ""


def needs_fbgemm_patches(device: int = 0) -> bool:
    """Return True if fbgemm_gpu ops are known to crash on this GPU."""
    return get_rocm_arch(device) in _ARCHS_NEEDING_FBGEMM_PATCHES


def needs_tbe_bypass(device: int = 0) -> bool:
    """Return True if TBE (SplitTableBatchedEmbeddingBagsCodegen) deadlocks."""
    return get_rocm_arch(device) in _ARCHS_NEEDING_TBE_BYPASS


# ---------------------------------------------------------------------------
# fbgemm operator patches
# ---------------------------------------------------------------------------

def apply_rocm_fbgemm_patches() -> None:
    """Apply ROCm compatibility patches for fbgemm_gpu operators.

    On certain AMD GPU architectures (currently gfx950 / MI355X), several
    fbgemm_gpu operators crash with SIGSEGV or deadlock.  We replace them with
    pure-PyTorch equivalents at runtime.

    For other ROCm architectures (e.g. gfx942 / MI300X) the native fbgemm ops
    are used unchanged because they work correctly on those GPUs.

    Call this AFTER importing torchrec / fbgemm_gpu.
    """
    if not torch.version.hip:
        return  # NVIDIA CUDA path – nothing to do

    arch = get_rocm_arch()
    fbgemm_loaded = hasattr(torch.ops, "fbgemm")

    if fbgemm_loaded and arch and arch not in _ARCHS_NEEDING_FBGEMM_PATCHES:
        # This architecture is expected to work fine with native fbgemm ops.
        return

    # If fbgemm is not loaded at all (e.g. .so link error), we still need to
    # register fallback ops so downstream code doesn't crash.
    if not fbgemm_loaded:
        import logging
        logging.getLogger(__name__).warning(
            "[ROCm] fbgemm_gpu failed to load; registering pure-PyTorch fallbacks"
        )

    def _safe_complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
        """Complete cumsum: [0, l0, l0+l1, ...], length n+1."""
        zeros = lengths.new_zeros(1)
        return torch.cat([zeros, torch.cumsum(lengths, dim=0)])

    def _safe_inclusive_cumsum(lengths: torch.Tensor) -> torch.Tensor:
        """Inclusive cumsum: [l0, l0+l1, ...], length n."""
        return torch.cumsum(lengths, dim=0)

    def _safe_exclusive_cumsum(lengths: torch.Tensor) -> torch.Tensor:
        """Exclusive cumsum: [0, l0, l0+l1, ...], length n (drops last)."""
        return torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, dim=0)[:-1]])

    def _safe_jagged_to_padded_dense(
        values: torch.Tensor,
        offsets: list,
        max_lengths: list,
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        """Convert jagged tensor to padded dense tensor."""
        off = offsets[0]  # 1D offsets tensor
        B = off.numel() - 1
        max_len = max_lengths[0] if max_lengths else int((off[-1]).item())
        D = values.shape[1] if values.dim() > 1 else 1
        if values.dim() == 1:
            out = values.new_full((B, max_len), padding_value)
            for i in range(B):
                s, e = off[i].item(), off[i + 1].item()
                l = min(e - s, max_len)
                out[i, :l] = values[s:s+l]
        else:
            out = values.new_full((B, max_len, D), padding_value)
            for i in range(B):
                s, e = off[i].item(), off[i + 1].item()
                l = min(e - s, max_len)
                out[i, :l] = values[s:s+l]
        return out

    def _safe_dense_to_jagged(
        dense: torch.Tensor,
        offsets: list,
        total_L: int = -1,
    ):
        """Convert padded dense tensor to jagged tensor.
        Returns a tuple (values, lengths) matching fbgemm's API.
        dense: [B, S] or [B, S, D]
        offsets: list of 1 tensor of shape [B+1]
        """
        off = offsets[0]
        B = off.numel() - 1
        if total_L < 0:
            total_L = int(off[-1].item())

        lengths = off[1:] - off[:-1]

        if dense.dim() == 3:
            D = dense.shape[2]
            out = dense.new_zeros((total_L, D))
            for i in range(B):
                s, e = off[i].item(), off[i + 1].item()
                l = e - s
                out[s:e] = dense[i, :l]
        elif dense.dim() == 2:
            out = dense.new_zeros((total_L,))
            for i in range(B):
                s, e = off[i].item(), off[i + 1].item()
                l = e - s
                out[s:e] = dense[i, :l]
        else:
            out = dense.new_zeros((total_L,))
        return (out, lengths)

    # Apply patches for all operators known to crash on this architecture,
    # or register fallbacks if fbgemm_gpu failed to load entirely.
    import logging
    logging.getLogger(__name__).info(
        "[ROCm] Applying fbgemm operator patches for arch=%s", arch or "unknown"
    )
    patch_map = {
        "asynchronous_complete_cumsum": _safe_complete_cumsum,
        "asynchronous_inclusive_cumsum": _safe_inclusive_cumsum,
        "asynchronous_exclusive_cumsum": _safe_exclusive_cumsum,
        "jagged_to_padded_dense": _safe_jagged_to_padded_dense,
        "dense_to_jagged": _safe_dense_to_jagged,
    }
    if not fbgemm_loaded:
        # fbgemm_gpu couldn't load at all — create a minimal stub namespace so
        # downstream torch.ops.fbgemm.* calls work via the fallbacks.
        class _FbgemmStub:
            pass
        stub = _FbgemmStub()
        for op_name, impl in patch_map.items():
            setattr(stub, op_name, impl)
        try:
            torch.ops.fbgemm = stub  # type: ignore[assignment]
        except Exception:
            pass
    else:
        for op_name, impl in patch_map.items():
            try:
                if hasattr(torch.ops.fbgemm, op_name):
                    setattr(torch.ops.fbgemm, op_name, impl)
            except Exception:
                pass

    # Patch torchrec's internal _to_offsets which calls asynchronous_complete_cumsum
    try:
        import torchrec.sparse.jagged_tensor as jt_module
        jt_module._to_offsets = _safe_complete_cumsum
    except Exception:
        pass


def initialize_single_rank():
    if torch.distributed.is_initialized():
        return
    torch.set_printoptions(precision=6, sci_mode=False)
    rank = 0
    device: torch.device = torch.device(f"cuda:{rank}")
    backend = "nccl"
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend=backend, init_method="tcp://127.0.0.1:12345", rank=rank, world_size=1
    )


def initialize_distributed():
    if torch.distributed.is_initialized():
        return
    torch.set_printoptions(precision=8, sci_mode=False)
    rank = int(os.environ["LOCAL_RANK"])
    device: torch.device = torch.device(f"cuda:{rank}")
    backend = "nccl"
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend=backend)
    # Apply ROCm patches after all imports and before any fbgemm ops are called
    apply_rocm_fbgemm_patches()


def initialize_model_parallel(tensor_model_parallel_size=1):
    if parallel_state.model_parallel_is_initialized():
        return
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size,
    )
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])


def destroy_global_state():
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    # TODO, find the reason why destroying pg hit nccl error when tpsize > 1
    if parallel_state.model_parallel_is_initialized():
        if parallel_state.get_tensor_model_parallel_world_size() == 1:
            torch.distributed.destroy_process_group(
                group=parallel_state.get_tensor_model_parallel_group()
            )
            torch.distributed.destroy_process_group(
                group=parallel_state.get_data_parallel_group(with_context_parallel=True)
            )
        parallel_state.destroy_model_parallel()
    torch.cuda.empty_cache()
    gc.collect()


@contextmanager
def auto_destroy_global_state():
    try:
        initialize_distributed()
        yield
    finally:
        destroy_global_state()


def set_random_seed(seed_):
    if not parallel_state.model_parallel_is_initialized():
        initialize_model_parallel()
    import random

    import numpy as np

    """Set random seed for reproducibility."""
    if seed_ is not None and seed_ > 0:
        # Only CP/TP ranks share the same seed
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * parallel_state.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        seed = seed + (10 * parallel_state.get_data_parallel_rank())

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)

            # We must maintain an rng state for torchrec, because with different world size, the state evolution differ
            # guarantee randomness across DPxTPxCPxPP for embedding-group
            seed = seed + 1234
            seed = seed + (1000 * parallel_state.get_context_parallel_rank())
            seed = seed + (10000 * parallel_state.get_tensor_model_parallel_rank())
            rng_tracker = tensor_parallel.get_cuda_rng_tracker()
            rng_tracker.add("sharded-embedding-group-seed", seed)

    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))
