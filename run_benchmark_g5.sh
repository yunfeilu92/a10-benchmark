#!/bin/bash
set -ex

echo "=== G5 (8x A10G) Benchmark ==="
echo "Instance: g5.48xlarge | Region: us-east-1"
echo "Start time: $(date)"

# Activate venv
source /home/ubuntu/pluto-env/bin/activate

# Set paths - all on NVMe
export NUPLAN_DATA_ROOT=/nuplan/dataset
export NUPLAN_MAPS_ROOT=/nuplan/dataset/maps/nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/nuplan/exp

# Ensure pluto src is importable by Ray workers
export PYTHONPATH=/opt/dlami/nvme/pluto:$PYTHONPATH

# ============================================================
# NCCL 优化 - A10G PCIe 拓扑专用配置
# g5 无 NVLink，GPU 通过 PCIe Gen4 x16 互联
# A10G 是 Ampere 架构 (SM 8.6)，24GB VRAM
# ============================================================
export NCCL_ALGO=Ring                    # Ring 在 PCIe 拓扑下通常最优
export NCCL_PROTO=Simple                 # Simple 协议在 PCIe 下延迟更低
export NCCL_MIN_NCHANNELS=4              # 增加并行通信通道
export NCCL_MAX_NCHANNELS=12
export NCCL_BUFFSIZE=8388608             # 8MB buffer，改善 PCIe 吞吐
export NCCL_P2P_LEVEL=PHB                # PCIe Host Bridge 级别 P2P
export CUDA_DEVICE_MAX_CONNECTIONS=1     # 改善 compute/communication overlap

# PyTorch 性能优化
export TORCH_CUDNN_V8_API_ENABLED=1      # 启用 cuDNN v8 API
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 优化显存分配

# TF32 优化 - A10G Ampere 架构支持 TF32，可显著提升 matmul 性能
export NVIDIA_TF32_OVERRIDE=1            # 全局启用 TF32

# Ray worker GPU 可见性
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HYDRA_FULL_ERROR=1

# ============================================================
# A10G 训练参数调整（对比 H200 / L40S）:
#
# | 参数         | H200           | L40S (G6E)        | A10G (G5)           | 原因                          |
# |-------------|----------------|-------------------|---------------------|------------------------------|
# | batch_size  | 384 (48/卡)    | 384 (48/卡)       | 256 (32/卡)         | VRAM 24GB，需降低 batch       |
# | precision   | 32             | bf16-mixed        | bf16-mixed          | A10G 支持 BF16 (Ampere)       |
# | num_workers | 32             | 16                | 16                  | g5.48xlarge 192 vCPU          |
# | Ray workers | 40             | 32                | 32                  | g5.48xlarge 192 vCPU          |
# | epochs      | 25             | 3                 | 3                   | benchmark 只需要 3 epoch       |
# ============================================================

cd /opt/dlami/nvme/pluto

# Print GPU topology for reference
echo "=== GPU Topology ==="
nvidia-smi topo -m

echo "========================================="
echo "Step 1: Feature Cache"
echo "Start: $(date)"
echo "========================================="

python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_pluto \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=32 \
    2>&1 | tee /home/ubuntu/cache_g5.log

echo "Cache done: $(date)"

# Verify cache has data
CACHE_COUNT=$(find /nuplan/exp/cache_pluto -name "*.gz" -o -name "*.pkl" 2>/dev/null | wc -l)
echo "Cache files: $CACHE_COUNT"
if [ "$CACHE_COUNT" -eq 0 ]; then
    echo "ERROR: Cache is empty! Aborting training."
    exit 1
fi

echo "========================================="
echo "Step 2: Training on 8x A10G"
echo "Start: $(date)"
echo "========================================="

# Monitor GPU utilization in background
nvidia-smi dmon -s umt -d 10 > /home/ubuntu/gpu_monitor_g5.log 2>&1 &
GPU_MON_PID=$!

# 同时记录 NCCL 通信信息（首次运行建议开启，后续可关闭）
export NCCL_DEBUG=WARN

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_training.py \
    py_func=train +training=train_pluto \
    worker=single_machine_thread_pool worker.max_workers=16 \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_pluto \
    cache.use_cache_without_dataset=true \
    data_loader.params.batch_size=256 \
    data_loader.params.num_workers=16 \
    lr=1e-3 epochs=3 warmup_epochs=1 weight_decay=0.0001 \
    lightning.trainer.params.accelerator=gpu \
    lightning.trainer.params.devices=8 \
    lightning.trainer.params.strategy=ddp_find_unused_parameters_false \
    lightning.trainer.params.precision=bf16-mixed \
    wandb.mode=disabled \
    wandb.project=g5-benchmark \
    wandb.name=pluto-8xA10G \
    2>&1 | tee /home/ubuntu/train_g5.log

kill $GPU_MON_PID 2>/dev/null

echo "Training done: $(date)"
echo "========================================="
echo "=== GPU Monitor Summary ==="
tail -20 /home/ubuntu/gpu_monitor_g5.log

echo ""
echo "=== Benchmark Complete ==="
echo "Logs:"
echo "  Cache:  /home/ubuntu/cache_g5.log"
echo "  Train:  /home/ubuntu/train_g5.log"
echo "  GPU:    /home/ubuntu/gpu_monitor_g5.log"
