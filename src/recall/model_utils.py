import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

EMBED_DIM = 64

class TwoTowerModel(pl.LightningModule):
    def __init__(self, num_users, num_items, embed_dim=EMBED_DIM, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

    # -------- training with In-Batch Negatives --------
    def _batch_ce_loss(self, user_vec, item_vec):
        logits  = torch.matmul(user_vec, item_vec.T)              # (B,B)
        labels  = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, _):
        user_vec = self.user_emb(batch["user"])
        item_vec = self.item_emb(batch["pos_item"])
        loss     = self._batch_ce_loss(user_vec, item_vec)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        user_vec = self.user_emb(batch["user"])
        item_vec = self.item_emb(batch["pos_item"])
        loss     = self._batch_ce_loss(user_vec, item_vec)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        user_vec = self.user_emb(batch["user"])
        item_vec = self.item_emb(batch["pos_item"])
        loss     = self._batch_ce_loss(user_vec, item_vec)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
