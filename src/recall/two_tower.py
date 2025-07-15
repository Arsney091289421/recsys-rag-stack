import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import mlflow.pytorch
import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# ---- MLflow  ----
mlflow.set_tracking_uri("http://localhost:5500")  
mlflow.set_experiment("two_tower_recall")
mlflow.pytorch.autolog()

torch.set_float32_matmul_precision('high')

EMBED_DIM = 64
BATCH_SIZE = 256

if torch.cuda.is_available():
    free_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}, total VRAM: {free_mem:.1f} GB")
    if free_mem <= 8:
        print("Automatically set batch size to 256 for 8GB VRAM.")
        BATCH_SIZE = 256
    else:
        print("You can safely use larger batch size.")
    # Save 15% Vram to system
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

# ---- Two-Tower Model ----
class TwoTowerModel(pl.LightningModule):
    def __init__(self, num_users, num_items, embed_dim=EMBED_DIM, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_emb(user_ids)
        item_vec = self.item_emb(item_ids)
        return (user_vec * item_vec).sum(dim=1)

    def bpr_loss(self, user_vec, pos_item_vec, neg_item_vec):
        pos_scores = (user_vec * pos_item_vec).sum(dim=1)
        neg_scores = (user_vec * neg_item_vec).sum(dim=1)
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

    def training_step(self, batch, batch_idx):
        user_ids = batch["user"]
        pos_item_ids = batch["pos_item"]
        neg_item_ids = torch.randint(0, self.hparams.num_items, pos_item_ids.size(), device=self.device)

        user_vec = self.user_emb(user_ids)
        pos_item_vec = self.item_emb(pos_item_ids)
        neg_item_vec = self.item_emb(neg_item_ids)

        loss = self.bpr_loss(user_vec, pos_item_vec, neg_item_vec)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ---- Load data ----
def load_data(data_path="data/split/train.parquet"):
    print("Loading data...")
    df = pd.read_parquet(data_path)
    df["userid"] = df["user_id"].astype("category").cat.codes
    df["itemid"] = df["item_id"].astype("category").cat.codes

    num_users = df["userid"].nunique()
    num_items = df["itemid"].nunique()

    print(f"Loaded {len(df)} interactions, {num_users} users, {num_items} items.")
    return df, num_users, num_items

# ---- Main ----
def main():
    df, num_users, num_items = load_data()

    dataset = InteractionDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, persistent_workers=False)

    model = TwoTowerModel(num_users=num_users, num_items=num_items, embed_dim=EMBED_DIM)

    print("Starting training...")
    trainer = pl.Trainer(
        max_epochs=7,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=50,
        precision="16-mixed"
    )
    trainer.fit(model, loader)

    print("Saving item embeddings and checkpoint...")
    os.makedirs("outputs", exist_ok=True)
    npy_path = "outputs/item_embeddings.npy"
    faiss_path = "outputs/index.faiss"
    ckpt_path = "outputs/two_tower_checkpoint.pt"

    item_emb_weight = model.item_emb.weight.detach().cpu().numpy()
    np.save(npy_path, item_emb_weight)
    torch.save(model.state_dict(), ckpt_path)

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    faiss.normalize_L2(item_emb_weight)

    for i in tqdm(range(0, len(item_emb_weight), 10000), desc="Adding to FAISS"):
        end = min(i + 10000, len(item_emb_weight))
        index.add(item_emb_weight[i:end])

    faiss.write_index(index, faiss_path)

    print("Uploading artifacts to MLflow...")
    mlflow.log_artifact(npy_path)
    mlflow.log_artifact(faiss_path)
    mlflow.log_artifact(ckpt_path)

    print("All artifacts saved and logged to MLflow!")

if __name__ == "__main__":
    main()
