# src/data/prepare_data.py
import pandas as pd
import numpy as np
import os
import mlflow

mlflow.set_tracking_uri("http://localhost:5500")
mlflow.set_experiment("prepare_data")

def main():
    with mlflow.start_run(run_name="prepare_data"):
        # 路径
        raw_dir = "data/raw/"
        processed_dir = "data/processed/"
        split_dir = "data/split/"
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(split_dir, exist_ok=True)

        # 读取数据
        events = pd.read_csv(os.path.join(raw_dir, "events.csv"))
        items_part1 = pd.read_csv(os.path.join(raw_dir, "item_properties_part1.csv"))
        items_part2 = pd.read_csv(os.path.join(raw_dir, "item_properties_part2.csv"))
        categories = pd.read_csv(os.path.join(raw_dir, "category_tree.csv"))

        # 合并 item 属性
        items_all = pd.concat([items_part1, items_part2])
        items_pivot = items_all.sort_values("timestamp").drop_duplicates(subset=["itemid", "property"], keep="last")
        items_wide = items_pivot.pivot(index="itemid", columns="property", values="value").reset_index()

        # 可选：关联类别树 enrich
        if "categoryid" in items_wide.columns and "categoryid" in categories.columns:
            items_wide["categoryid"] = items_wide["categoryid"].astype(str)
            categories["categoryid"] = categories["categoryid"].astype(str)
            items_wide = items_wide.merge(categories, left_on="categoryid", right_on="categoryid", how="left")

        # 生成 users 表（这里只统计 user_id）
        users = events[["visitorid"]].drop_duplicates().rename(columns={"visitorid": "user_id"})

        # interactions 表
        interactions = events.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
        interactions = interactions[["user_id", "item_id", "event", "timestamp"]]
        interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], unit="ms")

        # 划分 train/valid/test
        split_time1 = interactions["timestamp"].quantile(0.7)
        split_time2 = interactions["timestamp"].quantile(0.85)

        train = interactions[interactions["timestamp"] <= split_time1]
        valid = interactions[(interactions["timestamp"] > split_time1) & (interactions["timestamp"] <= split_time2)]
        test = interactions[interactions["timestamp"] > split_time2]

        # 保存 parquet
        users.to_parquet(os.path.join(processed_dir, "users.parquet"))
        items_wide.to_parquet(os.path.join(processed_dir, "items.parquet"))
        train.to_parquet(os.path.join(split_dir, "train.parquet"))
        valid.to_parquet(os.path.join(split_dir, "valid.parquet"))
        test.to_parquet(os.path.join(split_dir, "test.parquet"))

        # MLflow artifacts
        mlflow.log_artifact(os.path.join(processed_dir, "users.parquet"))
        mlflow.log_artifact(os.path.join(processed_dir, "items.parquet"))
        mlflow.log_artifact(os.path.join(split_dir, "train.parquet"))
        mlflow.log_artifact(os.path.join(split_dir, "valid.parquet"))
        mlflow.log_artifact(os.path.join(split_dir, "test.parquet"))

        mlflow.log_metric("train_size", len(train))
        mlflow.log_metric("valid_size", len(valid))
        mlflow.log_metric("test_size", len(test))

if __name__ == "__main__":
    main()
