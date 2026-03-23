#!/bin/bash
set -ex

source /home/ubuntu/pluto-env/bin/activate

# Fix numpy version - need 1.24.x for compatibility with shapely/scipy/torch
pip install "numpy==1.24.4"

# Reinstall shapely against correct numpy
pip install "shapely==2.0.1" --force-reinstall

# Verify all imports work
python -c "
import numpy; print(f'numpy={numpy.__version__}')
import shapely; print(f'shapely={shapely.__version__}')
import scipy; print(f'scipy={scipy.__version__}')
import torch; print(f'torch={torch.__version__}, gpus={torch.cuda.device_count()}')
import natten; print(f'natten={natten.__version__}')
import cv2; print(f'cv2={cv2.__version__}')
"

# Test pluto import
export PYTHONPATH=/opt/dlami/nvme/pluto:$PYTHONPATH
python -c "
from src.models.pluto.pluto_model import PlanningModel
m = PlanningModel(
    dim=128, state_channel=6, polygon_channel=6, history_channel=9,
    history_steps=21, future_steps=80, encoder_depth=4, decoder_depth=4,
    drop_path=0.2, dropout=0.1, num_heads=4, num_modes=12,
    state_dropout=0.75, use_ego_history=False, state_attn_encoder=True,
    use_hidden_proj=False,
)
params = sum(p.numel() for p in m.parameters())
print(f'PLUTO_OK params={params:,}')
"

echo "=== ENV_READY ==="
