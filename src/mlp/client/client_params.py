# mlp/client/config.py
from __future__ import annotations

import json
import os
from pathlib import Path

# --- DATA PIPELINE (FISSI) ---
CLIP_MIN = 0.0
CLIP_MAX = 100.0

TEST_SIZE = 0.2
RANDOM_STATE = 42
SHUFFLE_SPLIT = False

# --- TRAINING DEFAULTS (overridabili da trial) ---
LOCAL_EPOCHS = 5
BATCH_SIZE = 16
LR = 0.00018014645042168243
WEIGHT_DECAY = 0.004280652956204336

# --- MODEL DEFAULTS (overridabili da trial) ---
HIDDEN_SIZES = [64, 32, 16]
DROPOUT = 0.44322985509221996