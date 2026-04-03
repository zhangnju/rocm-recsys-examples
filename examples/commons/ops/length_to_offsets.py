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
import torch

try:
    import fbgemm_gpu  # for asynchronous_complete_cumsum  # noqa: F401
    _FBGEMM_AVAILABLE = True
except (ImportError, OSError):
    _FBGEMM_AVAILABLE = False


def _fbgemm_ops_crash_on_this_gpu() -> bool:
    """Return True if fbgemm async cumsum ops are known to SIGSEGV on this GPU.

    Currently gfx950 (MI355X) crashes; gfx942 (MI300X) and NVIDIA are fine.
    Checked once at module import time to avoid per-call overhead.
    """
    if not bool(getattr(torch.version, "hip", False)):
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "").split(":")[0].strip()
        return arch in {"gfx950"}
    except Exception:
        return False


_FBGEMM_BROKEN = _fbgemm_ops_crash_on_this_gpu() if torch.cuda.is_available() else False
_USE_FBGEMM = _FBGEMM_AVAILABLE and not _FBGEMM_BROKEN


def _cumsum_complete(t: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch fallback: [0, t[0], t[0]+t[1], ...]  with a trailing total."""
    z = torch.zeros(1, dtype=t.dtype, device=t.device)
    return torch.cat([z, torch.cumsum(t, dim=0)])


def length_to_complete_offsets(length_tensor: torch.Tensor):
    if _USE_FBGEMM:
        return torch.ops.fbgemm.asynchronous_complete_cumsum(length_tensor)
    return _cumsum_complete(length_tensor)


def length_to_inclusive_offsets(length_tensor: torch.Tensor):
    if _USE_FBGEMM:
        return torch.ops.fbgemm.asynchronous_inclusive_cumsum(length_tensor)
    return torch.cumsum(length_tensor, dim=0)


def length_to_exclusive_offsets(length_tensor: torch.Tensor):
    if _USE_FBGEMM:
        return torch.ops.fbgemm.asynchronous_exclusive_cumsum(length_tensor)
    z = torch.zeros(1, dtype=length_tensor.dtype, device=length_tensor.device)
    return torch.cat([z, torch.cumsum(length_tensor, dim=0)[:-1]])
