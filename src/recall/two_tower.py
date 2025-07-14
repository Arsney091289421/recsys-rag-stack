import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import mlflow.pytorch
import faiss
import pandas as pd

from torch.utils.data import DataLoader, Dataset

mlflow.set_tracking_uri("http://localhost:5500")
mlflow.set_experiment("two_tower_recall")
mlflow.pytorch.autolog()

EMBED_DIM = 64
BATCH_SIZE = 2048

# -------- Dataset --------
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

# -------- Lightning Module --------
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
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# -------- Load Data --------
def load_data(data_path="data/split/train.parquet"):
    df = pd.read_parquet(data_path)
    df["userid"] = df["userid"].astype("category").cat.codes
    df["itemid"] = df["itemid"].astype("category").cat.codes

    num_users = df["userid"].nunique()
    num_items = df["itemid"].nunique()

    return df, num_users, num_items

# -------- Main --------
def main():
    df, num_users, num_items = load_data()

    dataset = InteractionDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = TwoTowerModel(num_users=num_users, num_items=num_items, embed_dim=EMBED_DIM)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=50
    )
    trainer.fit(model, loader)

    # save item embedding
    item_emb_weight = model.item_emb.weight.detach().cpu().numpy()
    os.makedirs("outputs", exist_ok=True)
    npy_path = "outputs/item_embeddings.npy"
    faiss_path = "outputs/index.faiss"
    torch.save(model.state_dict(), "outputs/two_tower_checkpoint.pt")

    # save numpy embedding
    import numpy as np
    np.save(npy_path, item_emb_weight)

    # build FAISS index
    index = faiss.IndexFlatIP(EMBED_DIM)
    faiss.normalize_L2(item_emb_weight)
    index.add(item_emb_weight)
    faiss.write_index(index, faiss_path)

    print("Saved item embeddings and FAISS index.")

if __name__ == "__main__":
    main()
