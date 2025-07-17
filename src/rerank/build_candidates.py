"""
Generate top-K candidate items per user using trained Two-Tower model.

Usage
-----
python -m rerank.build_candidates \
    --ckpt_path outputs/best-checkpoint-v7.ckpt \
    --top_k 300 \
    --outfile outputs/candidates.parquet

Assumptions
-----------
* user_encoder.parquet / item_encoder.parquet are in `outputs/`
* Train/validation/test parquet files are in `data/split/`
* TwoTowerModel definition lives in `src/recall/model_utils.py`

Output columns
--------------
userid (int32)  |  itemid (int32)  |  score (float32)
"""

# src/rerank/build_candidates.py

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm

from recall.model_utils import TwoTowerModel, EMBED_DIM  # adjust if needed

def load_encoders(enc_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    user_enc = pd.read_parquet(enc_dir / "user_encoder.parquet")
    item_enc = pd.read_parquet(enc_dir / "item_encoder.parquet")
    return user_enc, item_enc

def build_faiss(index_vectors: np.ndarray) -> faiss.Index:
    faiss.normalize_L2(index_vectors)
    index = faiss.IndexFlatIP(index_vectors.shape[1])
    index.add(index_vectors)
    return index

def main(args):
    out_dir = Path(args.outfile).with_suffix('')  # remove .parquet
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing per-batch files to directory: {out_dir}/")

    # --- load encoders -------------------------------------------------------
    user_enc, item_enc = load_encoders(Path("outputs"))
    num_users = user_enc["userid"].max() + 1
    num_items = item_enc["itemid"].max() + 1

    # --- load model ----------------------------------------------------------
    print(f"Loading Two-Tower checkpoint from {args.ckpt_path} …")
    model = TwoTowerModel.load_from_checkpoint(
        args.ckpt_path,
        num_users=num_users,
        num_items=num_items,
        embed_dim=EMBED_DIM,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- item embeddings + FAISS index ---------------------------------------
    with torch.no_grad():
        item_matrix = model.item_emb.weight.detach().cpu().numpy().astype(np.float32)
    faiss_index = build_faiss(item_matrix)

    # --- candidate generation per batch --------------------------------------
    all_user_ids = user_enc["userid"].unique()
    batch_size = 1024

    with torch.no_grad():
        for i in tqdm(range(0, len(all_user_ids), batch_size), desc="scoring"):
            batch_users = all_user_ids[i : i + batch_size]
            batch_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            user_vec = model.user_emb(batch_tensor).detach().cpu().numpy().astype(np.float32)
            faiss.normalize_L2(user_vec)
            D, I = faiss_index.search(user_vec, args.top_k)

            batch_rows = []
            for u_idx, u in enumerate(batch_users):
                for j in range(args.top_k):
                    batch_rows.append((u, int(I[u_idx, j]), float(D[u_idx, j])))

            batch_df = pd.DataFrame(batch_rows, columns=["userid", "itemid", "score"], dtype="float32")
            batch_df[["userid", "itemid"]] = batch_df[["userid", "itemid"]].astype("int32")

            batch_path = out_dir / f"part_{i//batch_size:04d}.parquet"
            batch_df.to_parquet(batch_path, engine="pyarrow", index=False, compression="snappy")

    print(f"Done! Candidates written to folder: {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build top-K candidates using Two-Tower model")
    parser.add_argument("--ckpt_path", required=True, help="Path to Two-Tower checkpoint (.ckpt)")
    parser.add_argument("--top_k", type=int, default=300, help="Top K items per user")
    parser.add_argument("--outfile", default="outputs/candidates_parts", help="Output directory (no .parquet)")
    args = parser.parse_args()

    main(args)
