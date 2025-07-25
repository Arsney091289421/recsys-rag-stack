# Work in progress
### Important Note for macOS Users

When running MLflow with Docker on macOS, you might encounter the following error during `mlflow.log_artifact()`:

```
OSError: [Errno 30] Read-only file system
```

#### Why does this happen?

On macOS, Docker Desktop uses a Linux VM and overlay filesystem. If you configure `MLFLOW_DEFAULT_ARTIFACT_ROOT` as an absolute local path (e.g., `/mlflow-artifacts`), MLflow client may mistakenly try to create directories directly on your macOS root filesystem, which is read-only in this context.

#### How to fix it 

1. **Always enable server-side artifact serving.**

   Update your `docker-compose.yml` or MLflow server command to include:

   ```
   --serve-artifacts
   --artifacts-destination /mlflow-artifacts
   ```

2. **Use a properly mounted and writable volume.**

   Example volume configuration in `docker-compose.yml`:

   ```yaml
   volumes:
     - ./mlruns:/mlflow-artifacts
     - ./mlflow.db:/mlflow/mlflow.db
   ```

3. **Set permissions on your local artifact directory:**

   ```bash
   mkdir -p mlruns
   chmod -R 777 mlruns
   ```

# Windows + NVIDIA GPU user

   ```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

---

### MySQL Volume Backup

Run:

```bash
./backup_mysql.sh
```

This safely stops Docker, then creates a timestamped `.tar.gz` backup in the `backup/` folder.

---

### 1. `src/data/prepare_data.py`

This script is used for **data preprocessing and splitting**.
It loads raw event and item property files, merges them to enrich item information, and generates user-item interaction logs.
The script then splits the interactions into **train**, **validation**, and **test** sets based on timestamp quantiles.

**Outputs:**

* Processed user and item feature parquet files
* Train, validation, and test parquet files for downstream modeling
* All processed files are also logged to MLflow as artifacts for reproducibility

---

### 2. `src/recall` scripts

* **`two_tower.py`**: Implements the Two-Tower recall model using PyTorch Lightning. It trains user and item embeddings using BPR loss. The script outputs the best checkpoint (based on validation loss), final embeddings, a FAISS index for retrieval, and registers the model to MLflow.

* **`model_utils.py`**: Contains reusable Two-Tower model definitions and helper utilities, including the val/test evaluation logic and encoder exporting. This module is imported by other scripts to keep the main training script clean and modular.

* **`evaluate_recall.py`**: Computes offline recall\@50 metric on the test set using the saved checkpoint. It logs the recall score back to MLflow under the same run for consistent tracking.

**Outputs:**

* Best checkpoint file (`.ckpt`)
* Item embedding numpy file
* FAISS index file
* User and item ID encoders
* Registered model in MLflow with performance metrics, including recall\@50

---

# LambdaRank Re-ranking

## Overview

This module applies LambdaRank-based re-ranking on top of candidate items generated by a two-tower recall model.

## Data Summary

- **Initial Candidates**: ~294 million samples (1.4M users × top-300 items per user)
- **Positive Labels**: Matched from `events.csv` where `event` is `addtocart` or `transaction`
- **Negative Sampling**: 5% random downsampling of all negatives
- **Final Dataset**: ~14.7 million samples

## Features

- Lightweight feature vector (7 dimensions), including:
  - User statistics: total views, total purchases, conversion rate
  - Item statistics: total views, total purchases, conversion rate
  - Recall model score

## Training

- **Model**: LightGBM with `lambdarank` objective
- **Grouping**: Samples grouped by user ID
- **Evaluation Metric**: NDCG@10
- **Early Stopping**: Triggered at iteration 44
- **Training Time**: ~1 minute on CPU

## Performance

- **Train NDCG@10**: 0.99997
- **Valid NDCG@10**: 0.99987
- The model is compact and converges quickly, suitable for real-time re-ranking in recommendation systems.

### Model Comparison: Two-Tower vs. GBT (LambdaRank)

| Model           | Valid Users | NDCG@10  | Role                                 |
|----------------|-------------|----------|--------------------------------------|
| Two-Tower       | 100         | 0.0163   | Large-scale recall (fast, coarse)    |
| GBT (LambdaRank)| 54          | 0.5789   | Fine-grained ranking (accurate, learned features) |
