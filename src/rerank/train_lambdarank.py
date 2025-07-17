# src/rerank/train_lambdarank.py

import pandas as pd
import lightgbm as lgb
from pathlib import Path
from tqdm import tqdm
import numpy as np
import mlflow
from mlflow import MlflowClient

# --------------------
NEG_SAMPLE_FRAC = 0.05   # Fraction of negative samples to keep
VALID_FRAC = 0.1         # Fraction of users for validation
SEED = 42
np.random.seed(SEED)
# --------------------

# Setup MLflow
mlflow.set_tracking_uri("http://localhost:5500")
mlflow.set_experiment("lambdarank_rerank")

with mlflow.start_run():

    # 1. Define feature columns
    feature_cols = [
        "score",
        "user_total_views", "user_total_buys", "user_buy_rate",
        "item_total_views", "item_total_buys", "item_buy_rate",
    ]

    data_dir = Path("data/processed/rank_labeled")
    files = sorted(data_dir.glob("rank_*.parquet"))
    print(f"Processing {len(files)} partitioned files...")

    X_list, y_list, uid_list = [], [], []

    # 2. Load and sample each file
    for f in tqdm(files, desc="Loading and sampling"):
        df = pd.read_parquet(f)

        pos_df = df[df.label == 1]
        neg_df = df[df.label == 0].sample(frac=NEG_SAMPLE_FRAC, random_state=SEED)
        df = pd.concat([pos_df, neg_df], ignore_index=True)

        X_list.append(df[feature_cols].astype("float32"))
        y_list.append(df["label"].astype("int8"))
        uid_list.append(df["userid"].astype("int32"))

        del df, pos_df, neg_df

    # 3. Concatenate all chunks
    X_all = pd.concat(X_list, ignore_index=True)
    y_all = pd.concat(y_list, ignore_index=True)
    userids = pd.concat(uid_list, ignore_index=True)
    print("Total samples after sampling:", len(X_all))

    # 4. Train/Validation split by user
    unique_users = userids.unique()
    valid_users = set(np.random.choice(unique_users, size=int(len(unique_users) * VALID_FRAC), replace=False))

    mask_valid = userids.isin(valid_users)
    train_idx = np.where(~mask_valid)[0]
    valid_idx = np.where(mask_valid)[0]

    # 5. Group vectors
    group_train = userids[~mask_valid].value_counts().sort_index().values
    group_valid = userids[mask_valid].value_counts().sort_index().values

    # 6. LightGBM dataset
    train_set = lgb.Dataset(X_all.iloc[train_idx], label=y_all.iloc[train_idx], group=group_train)
    valid_set = lgb.Dataset(X_all.iloc[valid_idx], label=y_all.iloc[valid_idx], group=group_valid, reference=train_set)

    # 7. Training parameters
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "seed": SEED,
        "verbose": -1,
    }

    # Log params to MLflow
    mlflow.log_param("neg_sample_frac", NEG_SAMPLE_FRAC)
    mlflow.log_param("valid_frac", VALID_FRAC)
    mlflow.log_params(params)

    # 8. Train
    print("Training started...")
    model = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=30)]
    )

    # Log metrics
    mlflow.log_metric("best_ndcg_10", model.best_score["valid"]["ndcg@10"])
    mlflow.log_metric("best_iteration", model.best_iteration)

    # 9. Save model and register
    model_path = "outputs/lambdarank_model.txt"
    model.save_model(model_path)
    mlflow.log_artifact(model_path)

    run_id = mlflow.active_run().info.run_id
    model_name = "LambdaRankRecSys"
    model_uri = f"runs:/{run_id}/{model_path}"

    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
        print(f"Created registered model: {model_name}")
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            print(f"Model {model_name} already exists.")
        else:
            raise

    model_version = client.create_model_version(model_name, model_uri, run_id)
    print(f"Model registered successfully as {model_name}, version {model_version.version}")
