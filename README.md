# FedLIME: Federated Learning with LIME Explanations

FedLIME combines Federated Learning with LIME (Local Interpretable Model-agnostic Explanations) to provide interpretable machine learning in distributed settings. It trains models across multiple clients, then aggregates local LIME explanations to understand global feature importance while preserving privacy through Local Differential Privacy.

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Single Experiment

**Manual Implementation:**
```bash
python fedlime_newaggmethod.py \
    --dataset_name adult \
    --num_clients 3 \
    --communication_rounds 50 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --lime_action all \
    --lime_instances 100 \
    --lime_noise_std 0.01 \
    --out_dir my_experiment
```

**Flower Framework:**
```bash
python fedlime_flower.py \
    --dataset_name adult \
    --num_clients 3 \
    --num_rounds 50 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --lime_action all \
    --lime_instances 100 \
    --lime_noise_std 0.01 \
    --out_dir my_experiment_flower
```

**Centralized Baseline:**
```bash
python global_model_only.py \
    --dataset_name adult \
    --epochs 750 \
    --batch_size 128 \
    --lr 0.001 \
    --run_lime yes
```

### Run All Experiments

**Manual Implementation (9 federated + 3 centralized):**
```bash
./run_all_experiments.sh
```

**Flower Framework (9 federated):**
```bash
./run_flower_experiments.sh
```

## Visualize Results

```bash
python visualize_lime_results.py --result_dir results/adult/iid_adult_3clients_ldp --save
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset_name` | Dataset to use (adult, bank, law, etc.) | bank |
| `--num_clients` | Number of federated clients | 3 |
| `--num_rounds` / `--communication_rounds` | FL rounds | 5 / 50 |
| `--epochs` | Local training epochs per round | 2 / 15 |
| `--lime_action` | LIME analysis (none, all, rank, bins) | all |
| `--lime_instances` | Instances to explain per client | 100 |
| `--lime_noise_std` | Local DP noise std deviation | 0.0 |
| `--out_dir` | Output directory name | - |

## Datasets

The project supports three datasets:
- **Adult Census**: Income prediction
- **Bank Marketing**: Marketing campaign success
- **Law School**: Bar passage prediction

Datasets are located in the `datasets/` directory.

## Output Files

Each experiment creates:
- `acc_curve.npy` - Accuracy over rounds
- `bal_acc_curve.npy` - Balanced accuracy over rounds
- `auprc_curve.npy` - AUPRC over rounds
- `lime_global_mean_abs.npy` - Global feature importance
- `lime_client_mean_abs.npy` - Per-client feature importance
- `lime_similarity.npy` - Client similarity matrix
- `lime_client_fidelity.npy` - Per-client fidelity scores
- `lime_global_fidelity.npy` - Global fidelity score
- `lime_feature_ranking.csv` - Top-K features (human-readable)
- `lime_feature_bins.csv` - Feature importance bins
- `plots/` - Visualizations (if generated)

