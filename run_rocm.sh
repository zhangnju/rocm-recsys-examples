#!/bin/bash
# Launch script for running examples on AMD MI355X (gfx950, ROCm 7.2)
# Usage: ./run_rocm.sh [hstu|sid_gr] [PORT]

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE="${1:-hstu}"
PORT="${2:-6000}"

build_cuda_ops() {
    echo "=== Building hstu_cuda_ops for ROCm ==="
    cd "$REPO_ROOT/examples/commons"
    python setup_rocm.py build_ext --inplace
    cp hstu_cuda_ops*.so /opt/venv/lib/python3.12/site-packages/ 2>/dev/null || \
        echo "Note: hstu_cuda_ops installed in-place at examples/commons/"
    cd "$REPO_ROOT"
}

run_hstu() {
    echo "=== Running HSTU Ranking Training on MI355X ==="
    cd "$REPO_ROOT/examples/hstu"
    PYTHONPATH="$REPO_ROOT/examples/hstu:$REPO_ROOT/examples:$REPO_ROOT/examples/commons" \
        torchrun --nproc_per_node 1 --master_addr localhost --master_port "$PORT" \
        ./training/pretrain_gr_ranking.py \
        --gin-config-file ./training/configs/rocm_ranking.gin
}

run_sid_gr() {
    echo "=== Running SID-GR Training on MI355X ==="
    cd "$REPO_ROOT/examples/sid_gr"
    PYTHONPATH="$REPO_ROOT/examples/sid_gr:$REPO_ROOT/examples:$REPO_ROOT/examples/commons" \
        torchrun --nproc_per_node 1 --master_addr localhost --master_port "$PORT" \
        ./training/pretrain_sid_gr.py \
        --gin-config-file ./configs/sid_rocm.gin
}

# Always build the CUDA ops first
build_cuda_ops

case "$EXAMPLE" in
    hstu)
        run_hstu
        ;;
    sid_gr)
        run_sid_gr
        ;;
    *)
        echo "Usage: $0 [hstu|sid_gr] [PORT]"
        exit 1
        ;;
esac
