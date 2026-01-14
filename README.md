# Sleep Quality MLP Regressor

A federated learning system for sleep quality prediction using Multi-Layer Perceptron (MLP) neural networks. This project implements a distributed machine learning approach where multiple clients train local models that are aggregated on a central server using Flower (Flwr), a federated learning framework.
This model was first created and tested along other models in the [Experiment Repostory](https://github.com/AlessandroTinaoui/CSI-prog4-esperimenti)
## Overview

This project predicts sleep quality scores using time-series features extracted from sleep-tracking data. It leverages federated learning to train models across multiple data silos (user groups) while maintaining data privacy. The system includes:

- **Preprocessing Pipeline**: Extracts time-series features and cleans data across user groups
- **Federated Learning**: Multi-client training with server-side aggregation using Flower
- **MLP Regressor**: PyTorch-based neural network for sleep quality prediction
- **Evaluation Metrics**: Tracks Mean Absolute Error (MAE) across training rounds

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the System](#running-the-system)
- [Code Overview](#code-overview)
- [Data Format](#data-format)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlessandroTinaoui/Sleep-Quality-MLP-Regressor.git
   cd Sleep-Quality-MLP-Regressor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
    python -m pip install -U pip
    python -m pip install -e .
   ```

   The main dependencies include:
   - `flwr`: Federated learning framework
   - `tensorflow>=2.10.0`: Deep learning library
   - `torch`: PyTorch for neural networks
   - `numpy`: Numerical computing
   - `pandas`: Data manipulation
   - `scikit-learn`: Machine learning utilities
   - `matplotlib`: Data visualization

4. **Verify installation** (optional)
   ```bash
   python3 -c "import flwr; import torch; import tensorflow; print('All dependencies installed successfully!')"
   ```

## Project Structure

```
Sleep-Quality-MLP-Regressor/
├── data/                          # Dataset directory
│   ├── x_test.csv                 # Test features (holdout set)
│   └── group{0-8}/                # 9 client groups with training data
│       └── dataset_user_*_train.csv
├── src/                           # Main source code
│   ├── mlp/                       # MLP model implementation
│   │   ├── model.py               # MLPRegressor neural network class
│   │   ├── data.py                # Data loading and preprocessing utilities
│   │   ├── client/                # Federated learning client
│   │   │   ├── client_app.py      # Flower client implementation
│   │   │   ├── client_params.py   # Client configuration
│   │   │   ├── run_all.py         # Script to spawn all clients
│   │   │   └── results/           # Client-side results
│   │   ├── server/                # Federated learning server
│   │   │   ├── server_flwr.py     # Flower server implementation
│   │   │   └── config.py          # Server configuration (holdout client, etc.)
│   ├── preprocessing/             # Data preprocessing module
│   │   ├── preprocess_global.py   # Main preprocessing pipeline
│   │   ├── extract_ts_features.py # Time-series feature extraction
│   │   └── clients_dataset/       # Preprocessed client data
│   └── run_global_mae.py          # Main orchestrator script
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
└── README.md                      # This file
```

## Running the System

### Global MAE

Run the complete federated learning pipeline:

```bash
python3 -m src.run_global_mae
```

This will:
1. Start the Flower server
2. Launch federated learning clients (one per group, excluding holdout set)
3. Run multiple training rounds with model aggregation
4. Generate MAE summaries in `train_logs/`
5. Calculate mean MAE across all 9 clients

### Individual Components

#### Preprocessing Only

Preprocess and prepare data for all client groups:

```bash
python3 -m preprocessing.preprocess_global
```

**Output**: Cleaned and feature-engineered datasets in `src/mlp/preprocessing/clients_dataset/`

#### Start Federated Server

```bash
python3 -m mlp.server.server_flwr
```

#### Launch Federated Clients

In separate terminals (after starting the server):

```bash
# Launch all clients automatically
python3 -m mlp.client.run_all

# Or launch a single client manually
python3 -m mlp.client.client_app <client_id> <path_to_csv>
```

### Configuration

Edit `src/mlp/server/config.py` to modify:
- `HOLDOUT_CID`: Client ID to exclude from training (reserved for testing)
- Number of training rounds
- Client learning rates
- Model architecture parameters

## Code Overview

### 1. **Model Architecture** (`src/mlp/model.py`)

```python
class MLPRegressor(nn.Module):
    """Multi-Layer Perceptron for sleep quality regression"""
```

- **Input**: Feature vectors (variable dimension based on preprocessing)
- **Hidden Layers**: Configurable with ReLU activation and dropout
- **Output**: Single continuous value (sleep quality score)

### 2. **Data Pipeline** (`src/mlp/data.py`)

Key functions:
- `load_csv_dataset()`: Load user sleep data from CSV
- `split_train_test()`: Create train/test splits
- `apply_standardization()`: Normalize features using mean/std
- `ensure_feature_order_and_fill()`: Handle missing features with zero-filling

### 3. **Preprocessing** (`src/preprocessing/preprocess_global.py`)

Features:
- **Data Cleaning**: IQR-based outlier removal, null handling
- **Time-Series Features**: Extraction of statistical features from sleep tracking
- **Label Filtering**: Remove zero-quality labels
- **Group Organization**: Organize data by user groups (clients)

Configuration class: `CleanConfigGlobal`

### 4. **Federated Learning Server** (`src/mlp/server/server_flwr.py`)

- Aggregates model updates from multiple clients
- Implements FedAvg (Federated Averaging) algorithm
- Tracks global metrics across rounds
- Handles holdout client exclusion

### 5. **Federated Learning Client** (`src/mlp/client/client_app.py`)

Extends `fl.client.NumPyClient`:
- Local model training on client data
- Sends model parameters to server
- Receives aggregated parameters
- Logs local metrics (loss, MAE)

### 6. **Orchestrator** (`src/run_global_mae.py`)

Coordinates the complete pipeline:
- Spawns server process
- Launches client processes with proper timing
- Collects results and generates MAE summaries
- Handles process cleanup and logging



## Contact
Alice Olivares: alice.olivares@mail.polimi.it

Giovanni Scirocco: giovanninunzio.scirocco@mail.polimi.it

Alessandro Tinaoui: alessandro.tinaoui@mail.polimi.it
