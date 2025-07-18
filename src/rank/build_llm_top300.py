# build_llm_top300.py
import os, json, random, pandas as pd
from tqdm import tqdm
from pathlib import Path
random.seed(42)

SRC_DIR   = Path("data/processed/rank_gbm_scored")
OUT_JSONL = Path("data/processed/llm_top300.jsonl")
NEG_PER_POS = 5            # 1:5 neg sampling

os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

def make_prompt(r):
    return (f"The user viewed {r.user_total_views:.0f} items and purchased "
            f"{r.user_total_buys:.0f} (buy-rate {r.user_buy_rate:.2f}). "
            f"The candidate item was viewed {r.item_total_views:.0f} times "
            f"with buy-rate {r.item_buy_rate:.2f}. Two-Tower score "
            f"{r.score:.3f}, GBM score {r.gbm_score:.3f}.")

with open(OUT_JSONL, "w", encoding="utf-8") as fout:
    for fn in tqdm(sorted(os.listdir(SRC_DIR)), desc="writing jsonl"):
        df = pd.read_parquet(os.path.join(SRC_DIR, fn))

        for uid, grp in df.groupby("userid"):
            pos = grp[grp.label == 1]
            if pos.empty:
                continue

            # pos 
            for _, r in pos.iterrows():
                fout.write(json.dumps({
                    "userid": int(r.userid),
                    "itemid": int(r.itemid),
                    "input" : make_prompt(r),
                    "label" : 1}) + "\n")

            # neg random 5×
            need = len(pos) * NEG_PER_POS
            neg_candidates = grp[grp.label == 0]
            neg = neg_candidates.sample(min(need, len(neg_candidates)),
                                        random_state=42)
            for _, r in neg.iterrows():
                fout.write(json.dumps({
                    "userid": int(r.userid),
                    "itemid": int(r.itemid),
                    "input" : make_prompt(r),
                    "label" : 0}) + "\n")

print("Finished! JSONL saved at", OUT_JSONL)
