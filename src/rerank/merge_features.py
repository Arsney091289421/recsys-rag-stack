# src/rerank/merge_features.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# === Set paths ===
candidates_dir = Path("outputs/candidates_parts")
output_dir = Path("data/processed/rank_chunks")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading feature files...")
user_feats = pd.read_csv("data/features/user_stats.csv").rename(columns={"visitorid": "userid"})
item_feats = pd.read_csv("data/features/item_stats.csv")

# Explicitly downcast to save memory (optional)
user_feats = user_feats.astype("float32")
item_feats = item_feats.astype("float32")

print("Merging each candidate chunk...")

for part_path in tqdm(sorted(candidates_dir.glob("part_*.parquet"))):
    df = pd.read_parquet(part_path)

    # Merge features
    df = df.merge(user_feats, on="userid", how="left")
    df = df.merge(item_feats, on="itemid", how="left")

    # Output merged result
    out_path = output_dir / part_path.name.replace("part_", "rank_")
    df.to_parquet(out_path, index=False, compression="snappy")

print(f"All features merged and saved to: {output_dir}")
