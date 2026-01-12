# mlp/server/config.py
from __future__ import annotations

import json
import os
from pathlib import Path

# --- networking / fl ---
SERVER_ADDRESS = "127.0.0.1:8080"

HOLDOUT_CID = 2
# --- default federated params (overridabili da trial) ---
NUM_ROUNDS = 87
FRACTION_FIT = 1.0
FRACTION_EVALUATE = 1.0
MIN_FIT_CLIENTS = 8
MIN_EVALUATE_CLIENTS = 8
MIN_AVAILABLE_CLIENTS = 8
TOP_K_FEATURES = 80
MIN_STD_FEATURE_FS = 1e-8


# --- output ---
RESULTS_DIRNAME = "results"
GLOBAL_FEATURES_JSON = "global_features.json"
GLOBAL_SCALER_JSON = "global_scaler.json"
GLOBAL_MODEL_PTH = "global_model.pth"


