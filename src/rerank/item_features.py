# src/rerank/item_features.py
import pandas as pd
import os

df = pd.read_csv("data/raw/events.csv")
views = df[df["event"] == "view"]
buys = df[df["event"] == "transaction"]
os.makedirs("data/features", exist_ok=True)

# item view/buys 
item_stats = views.groupby("itemid").size().reset_index(name="item_total_views")
item_stats["item_total_buys"] = buys.groupby("itemid").size()
item_stats["item_total_buys"] = item_stats["item_total_buys"].fillna(0)
item_stats["item_buy_rate"] = item_stats["item_total_buys"] / item_stats["item_total_views"]

# save
item_stats.to_csv("data/features/item_stats.csv", index=False)
print("item_stats.csv saved.")