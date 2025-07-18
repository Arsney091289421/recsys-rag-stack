#src/rank/generate_gbm_scores.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb

# === Configure paths ===
input_dir = Path("data/processed/rank_labeled")
output_dir = Path("data/processed/rank_gbm_scored")
model_path = Path("outputs/lambdarank_model.txt")

output_dir.mkdir(parents=True, exist_ok=True)

# === Load LightGBM model ===
print(f"Loading LambdaRank model from: {model_path}")
model = lgb.Booster(model_file=str(model_path))

# === Feature columns (must match training time) ===
feature_cols = [
    "score", 
    "user_total_views", "user_total_buys", "user_buy_rate",
    "item_total_views", "item_total_buys", "item_buy_rate"
]

# === Process each parquet file ===
for file in tqdm(sorted(input_dir.glob("rank_*.parquet"))):
    df = pd.read_parquet(file)

    # Fill NaNs to avoid prediction errors
    df[feature_cols] = df[feature_cols].fillna(0)

    # Predict gbm_score
    df["gbm_score"] = model.predict(df[feature_cols])

    # Sort by userid + gbm_score descending order
    df = df.sort_values(by=["userid", "gbm_score"], ascending=[True, False])

    # Save to new directory, keeping the same file name
    output_path = output_dir / file.name
    df.to_parquet(output_path, index=False, compression="snappy")

print(f"\nAll files processed and saved to: {output_dir}")
