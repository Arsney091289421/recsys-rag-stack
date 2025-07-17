# src/rerank/user_stats.py
import pandas as pd
import os

df = pd.read_csv("data/raw/events.csv")
views = df[df["event"] == "view"]
buys = df[df["event"] == "transaction"]
os.makedirs("data/features", exist_ok=True)

# user view and buys
user_stats = views.groupby("visitorid").size().reset_index(name="user_total_views")
user_stats["user_total_buys"] = buys.groupby("visitorid").size()
user_stats["user_total_buys"] = user_stats["user_total_buys"].fillna(0)
user_stats["user_buy_rate"] = user_stats["user_total_buys"] / user_stats["user_total_views"]

# save
user_stats.to_csv("data/features/user_stats.csv", index=False)
print("user_stats.csv saved.")
