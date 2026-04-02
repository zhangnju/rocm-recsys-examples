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

# pyre-strict
from typing import Any, Dict, Tuple, Union

import torch
import torch.distributed as dist
import torchrec

# import our own finalize model grads
from commons.distributed.finalize_model_grads import finalize_model_grads
from commons.modules.embedding import DataParallelEmbeddingCollection
from commons.optimizer import OptimizerParam
try:
    from dynamicemb import DynamicEmbTableOptions
    from dynamicemb.get_planner import get_planner
    from dynamicemb.planner import (
        DynamicEmbeddingShardingPlanner as DynamicEmbeddingShardingPlanner,
    )
    from dynamicemb.shard import (
        DynamicEmbeddingBagCollectionSharder,
        DynamicEmbeddingCollectionSharder,
    )
    from dynamicemb.utils import TORCHREC_TYPES
    _DYNAMICEMB_AVAILABLE = True
except ModuleNotFoundError:
    DynamicEmbTableOptions = None  # type: ignore[assignment,misc]
    get_planner = None  # type: ignore[assignment]
    DynamicEmbeddingShardingPlanner = None  # type: ignore[assignment,misc]
    DynamicEmbeddingBagCollectionSharder = None  # type: ignore[assignment,misc]
    DynamicEmbeddingCollectionSharder = None  # type: ignore[assignment,misc]
    TORCHREC_TYPES = None  # type: ignore[assignment]
    _DYNAMICEMB_AVAILABLE = False
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from torch import distributed as dist
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)

# from torchrec.distributed import ModuleShardingPlan
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ShardedTensor, ShardingEnv
from torchrec.optim.optimizers import in_backward_optimizer_filter

DATA_PARALLEL_EMBEDDING_MODULE_NAME = "_data_parallel_embedding_collection"
from megatron.core import parallel_state


def apply_megatron_ddp(
    model: Union[DistributedModelParallel, torch.nn.Module],
    config: TransformerConfig,
    dense_optimizer_param: OptimizerParam,
    device: torch.device,
):
    """
    Apply megatron DDP to the model.
    If the original model is a DistributedModelParallel, the model._dmp_wrapped_module will be wrapped by DDP.
    Otherwise the original model will be wrapped by DDP.

    The original model is returned.
    """
    original_model = model
    if isinstance(model, DistributedModelParallel):
        model = original_model._dmp_wrapped_module
    else:
        model = original_model
    model = model.to(device)
    if config.fp16 or config.bf16:
        model = Float16Module(config, model)

    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
        check_for_nan_in_grad=False,
        bucket_size=True,
    )
    # MCORE DDP does not broadcast parameters implicitly
    if isinstance(original_model, DistributedModelParallel):
        original_model._dmp_wrapped_module = DDP(
            config,
            ddp_config,
            model,
        )
    else:
        original_model = DDP(
            config,
            ddp_config,
            model,
        )

    # only broadcast parameters within DataParallel group, TP group weights are initialized with the same rng states!
    def broadcast_params_for_non_model_parallel_embedding_modules():
        data_parallel_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True
        )
        for p in model.parameters():
            if not isinstance(p, TableBatchedEmbeddingSlice):
                dist.broadcast(
                    p.data,
                    src=torch.distributed.get_global_rank(data_parallel_group, 0),
                    group=data_parallel_group,
                )

    broadcast_params_for_non_model_parallel_embedding_modules()
    config.finalize_model_grads_func = finalize_model_grads

    param_dtype = torch.float32
    if config.bf16:
        param_dtype = torch.bfloat16
    elif config.fp16:
        param_dtype = torch.float16

    dense_optimizer_config = OptimizerConfig(
        optimizer=dense_optimizer_param.optimizer_str,
        lr=dense_optimizer_param.learning_rate,
        adam_beta1=dense_optimizer_param.adam_beta1,
        adam_beta2=dense_optimizer_param.adam_beta2,
        adam_eps=dense_optimizer_param.adam_eps,
        params_dtype=param_dtype,
        bf16=config.bf16,
        fp16=config.fp16,
        weight_decay=dense_optimizer_param.weight_decay,
    )
    dense_optimizer = get_megatron_optimizer(
        dense_optimizer_config,
        [
            original_model._dmp_wrapped_module
            if isinstance(original_model, DistributedModelParallel)
            else original_model
        ],
    )
    return original_model, dense_optimizer


_optimizer_str_to_optim_type = {
    "adam": EmbOptimType.ADAM,
    "sgd": EmbOptimType.EXACT_SGD,
    "row_wise_adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
}


class _ROCmEmbeddingCollection(torch.nn.Module):
    """Pure nn.Embedding replacement for EmbeddingCollection.
    Avoids TBE (SplitTableBatchedEmbeddingBagsCodegen) on architectures where
    it is known to deadlock (currently gfx950 / MI355X)."""
    def __init__(self, configs, device):
        super().__init__()
        from torchrec.modules.embedding_configs import EmbeddingConfig as TRecEmbCfg
        self._configs = configs
        self._feature_to_table = {}
        for cfg in configs:
            for feat in cfg.feature_names:
                self._feature_to_table[feat] = cfg.name
        self.embeddings = torch.nn.ModuleDict({
            cfg.name: torch.nn.Embedding(cfg.num_embeddings, cfg.embedding_dim, device=device)
            for cfg in configs
        })

    def forward(self, features):
        from torchrec.sparse.jagged_tensor import JaggedTensor
        out_dict = {}
        keys = features.keys()
        num_features = len(keys)
        # In a KJT, lengths has shape [B * num_features] and offsets has shape [B * num_features + 1]
        # where B is the batch size. Feature i spans offsets[i*B : (i+1)*B+1].
        # Compute B from total lengths / num_features
        total_lengths = features.lengths()  # shape [B * num_features]
        B = total_lengths.numel() // num_features if num_features > 0 else 0
        offsets = features.offsets()  # shape [B * num_features + 1]
        values = features.values()

        for i, feat_key in enumerate(keys):
            table_name = self._feature_to_table.get(feat_key)
            if table_name and table_name in self.embeddings:
                # Feature i spans samples [i*B : (i+1)*B]
                feat_start_offset = offsets[i * B].item()
                feat_end_offset = offsets[(i + 1) * B].item()
                indices = values[feat_start_offset:feat_end_offset].long()
                vocab_size = self.embeddings[table_name].num_embeddings
                indices = indices.clamp(0, vocab_size - 1)
                emb = self.embeddings[table_name](indices)
                # lengths for this feature
                feat_lengths = total_lengths[i * B : (i + 1) * B]
                out_dict[feat_key] = JaggedTensor(
                    values=emb,
                    lengths=feat_lengths,
                )
        return out_dict

    def embedding_configs(self):
        return self._configs


def _apply_dmp_rocm_fallback(
    model: torch.nn.Module,
    device: torch.device,
) -> torch.nn.Module:
    """Materialize meta-device EmbeddingCollection with nn.Embedding.
    Bypasses TBE (SplitTableBatchedEmbeddingBagsCodegen) on architectures where
    TBE is known to deadlock (currently gfx950 / MI355X)."""
    from torchrec.modules.embedding_modules import EmbeddingCollection
    from commons.modules.embedding import ShardedEmbedding

    # Walk the model tree and replace EmbeddingCollection with ROCm-safe version
    def replace_ec_modules(parent, prefix=''):
        for child_name, child_module in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child_module, EmbeddingCollection):
                configs = child_module.embedding_configs()
                replacement = _ROCmEmbeddingCollection(configs, device)
                setattr(parent, child_name, replacement)
            else:
                replace_ec_modules(child_module, full_name)

    replace_ec_modules(model)

    # Materialize remaining meta-device parameters.
    # model.to_empty() properly handles moving from meta to real device without copying.
    # Our _ROCmEmbeddingCollection already has real parameters so to_empty() is a no-op for them.
    model = model.to_empty(device=device)

    # Initialize parameters that were just materialized from meta (they are uninitialized)
    def _init_weights(module):
        from torchrec.modules.embedding_modules import EmbeddingCollection
        if isinstance(module, _ROCmEmbeddingCollection):
            return  # already initialized
        for name, param in module.named_parameters(recurse=False):
            if param.device == device:
                with torch.no_grad():
                    torch.nn.init.normal_(param, std=0.02)

    model.apply(_init_weights)

    return model


def apply_dmp(
    model: torch.nn.Module,
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions],
    sparse_optimizer_param: OptimizerParam,
    pg: torch.distributed.ProcessGroup,
    device: torch.device,
    pipeline_type: str = "native",
):
    # On specific ROCm architectures (gfx950 / MI355X), TBE deadlocks.
    # Use a pure nn.Embedding fallback for those GPUs only.
    # Other ROCm architectures (e.g. gfx942 / MI300X) use TBE natively.
    from commons.utils.initialize import needs_tbe_bypass
    if needs_tbe_bypass(device.index if device.index is not None else 0):
        import logging
        logging.getLogger(__name__).info(
            "[ROCm] TBE bypass active for arch=%s (device=%s)",
            __import__("commons.utils.initialize", fromlist=["get_rocm_arch"]).get_rocm_arch(),
            device,
        )
        return _apply_dmp_rocm_fallback(model, device)

    enable_prefetch_pipeline = pipeline_type == "prefetch"
    assert (
        sparse_optimizer_param.optimizer_str in _optimizer_str_to_optim_type
    ), f"embedding optimizer only support {list(_optimizer_str_to_optim_type.keys())}"
    fused_params = {
        "optimizer": _optimizer_str_to_optim_type[sparse_optimizer_param.optimizer_str],
        "learning_rate": sparse_optimizer_param.learning_rate,
        "beta1": sparse_optimizer_param.adam_beta1,
        "beta2": sparse_optimizer_param.adam_beta2,
        "eps": sparse_optimizer_param.adam_eps,
        # 'weight_decay_mode' : WeightDecayMode.NONE,
        # TODO, expose below params to users
        "output_dtype": SparseType.FP32,
        # only when compute kernel is FUSED_UVM_CACHING or KEY_VALUE are the below params effective.
        "cache_precision": SparseType.FP32,
        "stochastic_rounding": False,
        "prefetch_pipeline": enable_prefetch_pipeline,
    }
    eb_configs = []
    data_parallel_embedding_table_names = []
    data_parallel_embedding_module_names = []
    for k, module in model.named_modules():
        if TORCHREC_TYPES is not None and type(module) in TORCHREC_TYPES:
            eb_configs.extend(module.embedding_configs())
            if DATA_PARALLEL_EMBEDDING_MODULE_NAME in k:
                data_parallel_embedding_module_names.append(k)
                for config in module.embedding_configs():
                    data_parallel_embedding_table_names.append(config.name)

    qcomm_codecs_registry = get_qcomm_codecs_registry(
        qcomms_config=QCommsConfig(
            forward_precision=CommType.FP32,
            backward_precision=CommType.FP32,
        )
    )
    if _DYNAMICEMB_AVAILABLE:
        planner = get_planner(
            eb_configs,
            set(data_parallel_embedding_table_names),
            dynamicemb_options_dict,
            device,
            pipeline_type,
        )
        sharders = [
            DynamicEmbeddingBagCollectionSharder(
                qcomm_codecs_registry=qcomm_codecs_registry,
                fused_params=fused_params,
            ),
            DynamicEmbeddingCollectionSharder(
                qcomm_codecs_registry=qcomm_codecs_registry,
                use_index_dedup=True,
                fused_params=fused_params,
            ),
        ]
    else:
        # Fallback: use standard TorchRec planner and sharders when dynamicemb is unavailable
        from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
        from torchrec.distributed.embedding import EmbeddingCollectionSharder
        from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=dist.get_world_size(pg),
                compute_device=device.type,
            ),
        )
        sharders = [
            EmbeddingBagCollectionSharder(
                qcomm_codecs_registry=qcomm_codecs_registry,
                fused_params=fused_params,
            ),
            EmbeddingCollectionSharder(
                qcomm_codecs_registry=qcomm_codecs_registry,
                fused_params=fused_params,
            ),
        ]
    plan = planner.collective_plan(model, sharders, pg)
    data_parallel_sharding_plans = []
    for data_parallel_embedding_module_name in data_parallel_embedding_module_names:
        data_parallel_sharding_plans.append(
            plan.plan.pop(data_parallel_embedding_module_name, None)
        )
    # Shard model, the seed is forked to ensure different random state across all ranks
    with tensor_parallel.get_cuda_rng_tracker().fork("sharded-embedding-group-seed"):
        model = DistributedModelParallel(
            module=model,
            env=ShardingEnv.from_process_group(pg),
            device=device,
            sharders=sharders,
            plan=plan,
            init_data_parallel=False,
        )

    # Create keyed optimizer
    non_fused_sparse_params = {}
    for k, v in in_backward_optimizer_filter(model.named_parameters()):
        if v.requires_grad:
            if isinstance(v, ShardedTensor):
                non_fused_sparse_params[k] = v
    assert len(non_fused_sparse_params) == 0, "non_fused_sparse_params should be empty"

    if len(data_parallel_sharding_plans) > 0:
        unwrapped_model = model.module
        for dp_module_name, dp_sharding_plan in zip(
            data_parallel_embedding_module_names, data_parallel_sharding_plans
        ):
            data_parallel_embedding_collection_father_module_name = (
                dp_module_name.split(".")[:-1]
            )
            father_module = unwrapped_model
            for name in data_parallel_embedding_collection_father_module_name:
                father_module = getattr(father_module, name)
            data_parallel_embedding_collection = getattr(
                father_module, DATA_PARALLEL_EMBEDDING_MODULE_NAME
            )
            setattr(
                father_module,
                DATA_PARALLEL_EMBEDDING_MODULE_NAME,
                DataParallelEmbeddingCollection(
                    data_parallel_embedding_collection,
                    dp_sharding_plan,
                    ShardingEnv.from_process_group(pg),
                    fused_params,
                    device,
                ),
            )
    return model


def make_optimizer_and_shard(
    model: torch.nn.Module,
    config: TransformerConfig,
    sparse_optimizer_param: OptimizerParam,
    dense_optimizer_param: OptimizerParam,
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions] = {},
    pipeline_type: str = "native",
    device: torch.device = None,
    pg: torch.distributed.ProcessGroup = None,
) -> Tuple[DistributedModelParallel, torch.optim.Optimizer]:
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if pg is None:
        pg = dist.group.WORLD

    model = apply_dmp(
        model,
        dynamicemb_options_dict,
        sparse_optimizer_param,
        pg,
        device,
        pipeline_type,
    )
    model, dense_optimizer = apply_megatron_ddp(
        model, config, dense_optimizer_param, device
    )

    return model, dense_optimizer
