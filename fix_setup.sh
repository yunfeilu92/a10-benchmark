#!/bin/bash
set -ex

source /home/ubuntu/pluto-env/bin/activate

# Fix: pin numpy to compatible version before installing nuplan-devkit
# nuplan-devkit requires old numpy but we need numpy that works with Python 3.10
pip install "numpy<2.0,>=1.23"

# Install Cython (needed for some nuplan deps)
pip install Cython

echo "=== INSTALL NATTEN ==="
pip install natten==0.17.5+torch250cu124 -f https://shi-labs.com/natten/wheels/ --trusted-host shi-labs.com || {
    echo "NATTEN prebuilt failed, trying source..."
    pip install natten==0.17.5 --no-build-isolation || echo "NATTEN_SKIPPED - will continue without it"
}

echo "=== INSTALL NUPLAN-DEVKIT ==="
cd /home/ubuntu
if [ ! -d nuplan-devkit ]; then
    git clone https://github.com/motional/nuplan-devkit.git
fi
cd nuplan-devkit

# Install deps first, then editable install
pip install -r ./requirements.txt || true
pip install -e . || true
echo "=== NUPLAN_DEVKIT_DONE ==="

echo "=== INSTALL PLUTO ==="
cd /home/ubuntu
if [ ! -d pluto ]; then
    git clone https://github.com/jchengai/pluto.git
fi
cd pluto
pip install -e . || true
pip install pytorch-lightning==2.0.1 hydra-core==1.3.2 wandb l5kit shapely==2.0.1 || true
echo "=== PLUTO_DONE ==="

echo "=== PATCHING BF16 ==="
cd /home/ubuntu/pluto
ENCODER_FILE="src/models/pluto/modules/agent_encoder.py"
if [ -f "$ENCODER_FILE" ]; then
    grep -q "dtype=x_agent_tmp.dtype" "$ENCODER_FILE" || \
    sed -i 's/torch\.zeros(num_agent_type, self\.d_model, device=x_agent_tmp\.device)/torch.zeros(num_agent_type, self.d_model, device=x_agent_tmp.device, dtype=x_agent_tmp.dtype)/g' "$ENCODER_FILE"
    echo "Patched $ENCODER_FILE"
fi
find src -type d -exec sh -c 'touch "$1/__init__.py" 2>/dev/null' _ {} \;

echo "=== COPY TO NVME ==="
df -h /opt/dlami/nvme 2>/dev/null || sudo mkdir -p /opt/dlami/nvme
sudo cp -r /home/ubuntu/pluto /opt/dlami/nvme/pluto 2>/dev/null || true
sudo cp -r /home/ubuntu/nuplan-devkit /opt/dlami/nvme/nuplan-devkit 2>/dev/null || true
sudo mkdir -p /opt/dlami/nvme/nuplan/dataset /opt/dlami/nvme/nuplan/exp
sudo chown -R ubuntu:ubuntu /opt/dlami/nvme/nuplan 2>/dev/null || true
sudo ln -sfn /opt/dlami/nvme/nuplan /nuplan

echo "=== FINAL VERIFY ==="
python -c "
import torch
print(f'PyTorch={torch.__version__}, GPUs={torch.cuda.device_count()}, CUDA={torch.version.cuda}')
print(f'BF16={torch.cuda.is_bf16_supported()}')
"
python -c "import natten; print(f'NATTEN={natten.__version__}')" 2>&1 || echo "NATTEN not available"
python -c "from nuplan.planning.training.preprocessing.features.raster import Raster; print('nuplan-devkit OK')" 2>&1 || echo "nuplan-devkit partial"
echo "=== ALL_DONE ==="
