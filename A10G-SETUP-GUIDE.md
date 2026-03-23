# AWS G5 (A10G) Benchmark 配置指南

## 实例规格 — g5.48xlarge

| 规格 | 数值 |
|------|------|
| GPU | 8x NVIDIA A10G |
| VRAM | 24 GB / GPU (192 GB total) |
| 架构 | Ampere (SM 8.6) |
| FP32 算力 | 31.2 TFLOPS / GPU |
| TF32 算力 | 125 TFLOPS / GPU |
| BF16 算力 | 125 TFLOPS / GPU (稀疏 250) |
| GPU 互联 | PCIe Gen4 x16 (无 NVLink) |
| vCPU | 192 |
| 内存 | 768 GB |
| 本地存储 | 2x NVMe SSD |
| 网络 | 100 Gbps |

## 与 H200 / L40S 对比

| 维度 | A10G (G5) | L40S (G6E) | H200 (P5en) |
|------|-----------|------------|-------------|
| VRAM | 24 GB | 48 GB | 80 GB |
| BF16 TFLOPS | 125 | 362 | 1979 |
| GPU 互联 | PCIe Gen4 | PCIe Gen4 | NVLink |
| 价格 (按需) | ~$16/h | ~$78/h | ~$99/h |
| 性价比定位 | 入门级推理/训练 | 中端训练 | 高端训练 |

## 启动实例 (us-east-1)

### 查询可用 AMI

```bash
aws ec2 describe-images \
    --region us-east-1 \
    --owners amazon \
    --filters "Name=name,Values=*Deep Learning AMI GPU PyTorch*Ubuntu*" \
    --query 'Images | sort_by(@, &CreationDate) | [-3:].[ImageId,Name,CreationDate]' \
    --output table
```

### 启动实例

```bash
aws ec2 run-instances \
    --region us-east-1 \
    --instance-type g5.48xlarge \
    --image-id ami-xxxxxxxxx \
    --key-name your-key \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=a10g-benchmark}]'
```

## 关键差异 & 调优要点

### 1. 显存限制 (24GB vs 80GB)
- Pluto 模型很小 (~2MB)，24GB 足够
- Batch size 降到 256 (32/卡)，比 H200 的 384 (48/卡) 小
- 如果 OOM，可进一步降到 192 (24/卡) 或 128 (16/卡)

### 2. PCIe 互联 (与 G6E 类似)
- NCCL 配置与 G6E 相同：Ring 算法 + Simple 协议
- DDP gradient sync 是瓶颈，但 Pluto 模型小，影响有限

### 3. TF32 加速
- A10G 支持 TF32（Ampere 架构特性）
- TF32 matmul 比 FP32 快 ~3x，接近 BF16 精度
- 设置 `NVIDIA_TF32_OVERRIDE=1` 全局启用

### 4. 算力预期
- A10G 单卡 TF32 算力 125 TFLOPS，8 卡合计 1000 TFLOPS
- 对比 H200 的 15832 TFLOPS (8卡) 和 L40S 的 2896 TFLOPS (8卡)
- 预期训练速度约为 L40S 的 1/3，H200 的 1/16

## 运行步骤

```bash
# 1. SSH 到实例
ssh -i your-key.pem ubuntu@<instance-ip>

# 2. 克隆 benchmark 仓库
git clone https://github.com/yunfeilu92/a10-benchmark.git
cd a10-benchmark

# 3. 环境配置
bash setup_g5.sh

# 4. 下载数据
source ~/pluto-env/bin/activate
bash download_data.sh

# 5. 运行 benchmark
bash run_benchmark_g5.sh

# 6. 运行 profiling (可选)
python profile_pluto.py
```

## OOM 故障排除

如果训练出现 CUDA OOM：

```bash
# 方案 1: 降低 batch size
# 修改 run_benchmark_g5.sh 中的 data_loader.params.batch_size
# 256 -> 192 -> 128

# 方案 2: 启用 gradient checkpointing (如果 Pluto 支持)
# lightning.trainer.params.gradient_clip_val=1.0

# 方案 3: 检查显存占用
nvidia-smi
```

## 预期结果

基于 H200 和 L40S 的数据外推：

| 指标 | A10G (预估) | L40S (实测) | H200 (实测) |
|------|-------------|-------------|-------------|
| Forward pass | ~200-300ms | ~100ms | ~71ms |
| Epoch 时间 | ~30-45 min | ~15 min | ~8 min |
| GPU 利用率 | ~30-40% | ~35% | ~35% |

> 注意：Pluto 模型很小 (dim=128)，无法充分利用任何高端 GPU 的算力。
> A10G 因为 SM 数量较少，理论上小模型的利用率可能反而更高。
