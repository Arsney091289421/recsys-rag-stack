import os
import torch
import pandas as pd
import numpy as np
import faiss
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl
from evaluate_recall import compute_recall_at_k,RecallEvalCallback
from model_utils import TwoTowerModel, EMBED_DIM

# ---- MLflow ----
mlflow.set_tracking_uri("http://localhost:5500")  
mlflow.set_experiment("two_tower_recall")
mlflow.pytorch.autolog(log_models=False)

torch.set_float32_matmul_precision('high')

BATCH_SIZE = 512

if torch.cuda.is_available():
    free_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}, total VRAM: {free_mem:.1f} GB")
    if free_mem <= 8:
        print("Automatically set batch size to 512 for 8GB VRAM.")
        BATCH_SIZE = 512
    else:
        print("You can safely use larger batch size.")
    torch.cuda.set_per_process_memory_fraction(0.85, 0)
    print("Set per-process GPU memory fraction to 85% to reserve ~1 GB for system.")
else:
    print("CUDA not available, will use CPU.")

# ---- Dataset ----
class InteractionDataset(Dataset):
    def __init__(self, interactions_df):
        self.users = interactions_df["userid"].values
        self.items = interactions_df["itemid"].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return {
            "user": self.users[idx],
            "pos_item": self.items[idx],
        }

# ---- Load data ----
def load_data(split="train"):
    print(f"Loading {split} data...")
    df = pd.read_parquet(f"data/split/{split}.parquet")

    if split == "train":
        # save encoder for training
        df["userid"] = df["user_id"].astype("category").cat.codes
        df["itemid"] = df["item_id"].astype("category").cat.codes

        user_encoder = df[["user_id", "userid"]].drop_duplicates()
        item_encoder = df[["item_id", "itemid"]].drop_duplicates()

        os.makedirs("outputs", exist_ok=True)
        user_encoder.to_parquet("outputs/user_encoder.parquet")
        item_encoder.to_parquet("outputs/item_encoder.parquet")
        print("Saved user and item encoders.")

    else:
        # load encoder for test and vaild
        user_encoder = pd.read_parquet("outputs/user_encoder.parquet")
        item_encoder = pd.read_parquet("outputs/item_encoder.parquet")

        df = df.merge(user_encoder, on="user_id", how="left")
        df = df.merge(item_encoder, on="item_id", how="left")

        df = df[~df["userid"].isna()]
        df = df[~df["itemid"].isna()]
        df["userid"] = df["userid"].astype(int)
        df["itemid"] = df["itemid"].astype(int)

    num_users = df["userid"].nunique()
    num_items = df["itemid"].nunique()

    print(f"Loaded {len(df)} interactions, {num_users} users, {num_items} items.")
    return df, num_users, num_items

# ---- Main ----
def main():
    # Load train
    train_df, num_users, num_items = load_data("train")
    val_df, _, _ = load_data("valid")
    test_df, _, _ = load_data("test")

    train_dataset = InteractionDataset(train_df)
    val_dataset = InteractionDataset(val_df)
    test_dataset = InteractionDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TwoTowerModel(num_users=num_users, num_items=num_items, embed_dim=EMBED_DIM)

    # define recall_callback 
    ckpt_path = "outputs/tmp-epoch-checkpoint.ckpt"
    recall_callback = RecallEvalCallback(
        ckpt_path=ckpt_path,
        k=50,
    )

    print("Starting training...")
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=50,
        precision="16-mixed",
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="outputs",
                filename="best-checkpoint",
                monitor="val_loss",
                save_top_k=1,
                mode="min"
            ),
            recall_callback  
        ]
    )
    trainer.fit(
       model,
       train_loader,
       val_loader,
     # ckpt_path="outputs/best-checkpointV*.ckpt"    Countinue training from existing checkpoints-choose archived model version
)

    print("Running test...")
    trainer.test(model, dataloaders=test_loader)


    # ---- Save embeddings and checkpoint ----
    print("Saving item embeddings and checkpoint...")
    npy_path = "outputs/item_embeddings.npy"
    faiss_path = "outputs/index.faiss"
    ckpt_path = "outputs/best-checkpoint.ckpt"

    item_emb_weight = model.item_emb.weight.detach().cpu().numpy()
    np.save(npy_path, item_emb_weight)

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    faiss.normalize_L2(item_emb_weight)

    for i in tqdm(range(0, len(item_emb_weight), 10000), desc="Adding to FAISS"):
        end = min(i + 10000, len(item_emb_weight))
        index.add(item_emb_weight[i:end])

    faiss.write_index(index, faiss_path)

    # ---- Upload artifacts ----
    print("Uploading artifacts to MLflow...")
    mlflow.log_artifact(npy_path)
    mlflow.log_artifact(faiss_path)
    mlflow.log_artifact("outputs/user_encoder.parquet")
    mlflow.log_artifact("outputs/item_encoder.parquet")
    mlflow.log_artifact(ckpt_path)
    print("All artifacts saved and logged to MLflow!")

    # ---- register model ----
    run_id = mlflow.active_run().info.run_id
    model_name = "TwoTowerRecSys"
    model_uri = f"runs:/{run_id}/{ckpt_path}"

    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
        print(f"Created new registered model: {model_name}")
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            print(f"Model {model_name} already exists.")
        else:
            raise

    model_version = client.create_model_version(model_name, model_uri, run_id)
    print("Model registered successfully!")
    print("Name:", model_name)
    print("Version:", model_version.version)

if __name__ == "__main__":
    main()
