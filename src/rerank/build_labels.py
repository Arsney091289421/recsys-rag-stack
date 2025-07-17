# src/rerank/build_labels.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Extract positive samples from user events (view, addtocart, transaction)
events = pd.read_csv("data/raw/events.csv")
positive_pairs = events[events["event"].isin(["transaction", "addtocart", "view"])]
positive_pairs = positive_pairs[["visitorid", "itemid"]].drop_duplicates()
positive_pairs.columns = ["userid", "itemid"]

print("Number of positive samples:", len(positive_pairs))

# Iterate through each rank_chunk and assign labels
input_dir = Path("data/processed/rank_chunks")
output_dir = Path("data/processed/rank_labeled")
output_dir.mkdir(parents=True, exist_ok=True)

for file in tqdm(sorted(input_dir.glob("rank_*.parquet"))):
    df = pd.read_parquet(file)

    # Merge with positive samples; matched pairs get label=1, others label=0
    df = df.merge(positive_pairs.assign(label=1), on=["userid", "itemid"], how="left")
    df["label"] = df["label"].fillna(0).astype("int8")

    df.to_parquet(output_dir / file.name, index=False, compression="snappy")

print("All candidate sets have been labeled and saved to:", output_dir)
