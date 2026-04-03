# Examples: to demonstrate how to do training and inference generative recommendation models

## Generative Recommender Introduction
Meta's paper ["Actions Speak Louder Than Words"](https://arxiv.org/abs/2402.17152) introduces a novel paradigm for recommendation systems called **Generative Recommenders(GRs)**, which reformulates recommendation tasks as generative modeling problems. The work introduced Hierarchical Sequential Transduction Units (HSTU), a novel architecture designed to handle high-cardinality, non-stationary data streams in large-scale recommendation systems. HSTU enables both retrieval and ranking tasks. As noted in the paper, “HSTU-based GRs, with 1.5 trillion parameters, improve metrics in online A/B tests by 12.4% and have been deployed on multiple surfaces of a large internet platform with billions of users.”

In this example, we introduce the model architecture, training, and inference processes of HSTU. For more details, refer to the [training](./training/) and [inference](./inference/) entry folders, which include comprehensive guides and benchmark results.

## Ranking Model Introduction
The model structure of the generative ranking model can be depicted by the following picture.
![ranking model structure](./figs/ranking_model_structure.png)

### Input
The input to the HSTU model consists solely of pure categorical features, and it does not accommodate numerical features. The model supports three types of tokens:
* Contextual Tokens: Represent the user side info.
* Item Tokens: Represent the items being recommended.
* Action Tokens: Optional. Represent user actions associated with these items. Please note that if a user has multiple actions associated with a single item token, these actions must be merged into a single token during data preprocessing. For further details, please refer to [the related issue](https://github.com/facebookresearch/generative-recommenders/issues/114).

It is crucial that the number of item tokens matches the number of action tokens. This alignment ensures that each item can be effectively paired with its corresponding user action, as the paper said.

### Embedding Table
The embedding mechanism includes three types of distinct tables:
* Contextual Embedding Table: Corresponds to contextual tokens.
* Item Embedding Table: Corresponds to item tokens.
* Action Embedding Table: Corresponds to action tokens if provided.

### HSTU Block
The HSTU block is a core component of the architecture, which modifies traditional attention mechanisms to effectively handle large, non-stationary vocabularies typical in recommendation systems. 
* **Preprocessing**: After retrieving the embedding vectors from the tables, the HSTU preprocessing stage follows. If action embeddings are provided, the model interleaves the item and action embedding vectors. It then concatenates the contextual embeddings with the interleaved item and action embeddings, ensuring that each sample starts with contextual embeddings followed by item and action sequence pairs. Finally, the model applies position encoding.

* **Postprocessing**: If candidate items are specified, the model predicts only these candidates by filtering candidate item embeddings in the postprocessing. Otherwise, all item embeddings will be selected to be used for prediction.

### Prediction Head
The prediction head of the HSTU model employs a MLP network structure, enabling multi-task predictions. 

## Running the examples

* [HSTU training example](./training/)
* [HSTU inference example](./inference/)

## ROCm / AMD GPU Support (MI355X / gfx950)

HSTU training has been ported to AMD Instinct MI355X (gfx950, ROCm 7.2). The following sections describe what works and the porting changes made.

### Quick Start (ROCm)

```bash
cd examples/hstu
PYTHONPATH=/path/to/repo/examples/hstu:/path/to/repo/examples:/path/to/repo/examples/commons \
  torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
  ./training/pretrain_gr_ranking.py --gin-config-file ./training/configs/rocm_ranking.gin
```

See [`training/configs/rocm_ranking.gin`](./training/configs/rocm_ranking.gin) for the ROCm-specific training config.

### ROCm Code Changes

#### 1. Triton Jagged Ops Fallback (`ops/triton_ops/triton_jagged.py`)
`concat_2D_jagged_w_prefix` and `split_2D_jagged_w_prefix` Triton kernels fail on the AMD Triton backend (`TritonAMDGPUCanonicalizePointers` PassManager error). Pure PyTorch fallbacks were added to `_Concat2DJaggedFunction` and `_Split2DJaggedFunction` (forward and backward) when `torch.version.hip` is set.

#### 2. fbgemm Cumsum Ops (`commons/ops/length_to_offsets.py`)
Several `fbgemm_gpu` asynchronous cumsum operators (`asynchronous_complete_cumsum`, etc.) cause SIGSEGV on gfx950. `length_to_offsets.py` now auto-detects gfx950 at import time and uses pure PyTorch `torch.cumsum` instead.

#### 3. HSTU Attention Configs (`ops/triton_ops/triton_hstu_attention.py`)
ROCm-specific Triton config parameters added (`USE_TLX=False`, `NUM_BUFFERS=1`, etc.). The invalid `causal` kwarg removed from `fused_hstu_op.py` calls.

#### 4. Training Config (`training/configs/rocm_ranking.gin`)
- `kernel_backend = 'triton'` (not `'cutlass'` — CUTLASS is NVIDIA-only)
- `pipeline_type = 'none'` (avoids TBE all-to-all which deadlocks on gfx950)
- `enable_balanced_shuffler = False` (avoids `keyed_jagged_index_select_dim1` SIGSEGV)

### Unit Test Status (ROCm)

Run tests with:
```bash
LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH \
PYTEST_FIRST_PARAM_ONLY=1 \
  torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
  -m pytest test/<test_file>.py -k "not CUTLASS"
```

| Status | Test Files |
|--------|-----------|
| ✅ Pass | `test_addmm`, `test_jagged_tensor`, `test_metrics`, `test_ln_silu`, `test_ln_mul_dropout`, `test_triton_silu`, `test_hstu_preprocess`, `test_hstu_op` (excl. CUTLASS), `test_hstu_layer` (excl. CUTLASS), `test_collective`, `test_position_encoder` |
| ⚠️ Skip (expected) | `test_checkpointing`, `test_pipeline` — requires `dynamicemb` (NVIDIA-only) or `Float16Module` (needs Transformer Engine); `test_embedding` — NVEMB backend is NVIDIA-only; CUTLASS parametrized cases |
| ❌ Cannot run | `test_kvcache`, `test_paged_*`, `test_hstu_block_inference` — require `paged_kvcache_ops`/nvcomp (NVIDIA-only); `test_batch_balancer` — `keyed_jagged_index_select_dim1` SIGSEGV on gfx950; `test_dataset` — requires real dataset files; `hstu_attn/` — requires `hstu_attn_varlen_func` CUDA binary |

Test infrastructure changes for ROCm:
- `test/conftest.py` — imports `fbgemm_gpu` first, then applies ROCm patches (order matters)
- `test_utils.py` — `dynamicemb` import wrapped in try/except with `pytest.skip`
- `test/test_checkpointing.py` — `test_data_parallel_embedding_collection` skipped on gfx950

### Inference Status (ROCm)

Inference scripts (`inference/`) are **not supported** on ROCm:
- `inference_dense_module.py` has an unconditional `import paged_kvcache_ops` which depends on `nvcomp` (NVIDIA-only)
- `inference_gr_ranking.py` additionally requires a real dataset and trained checkpoint

# Acknowledgements

We would like to thank Yueming Wang (yuemingw@meta.com) and Jiaqi Zhai(jiaqiz@meta.com) for their guidance and assistance with the paper Action Speaks Louder Than Words during our efforts to understand the algorithm and reproduce the results. We also extend our gratitude to all the authors of the paper for their contributions and guidance. In addition, we would like to express special thanks to developers of [generative-recommenders](https://github.com/facebookresearch/generative-recommenders) that we have referenced. 
