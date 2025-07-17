import torch, faiss, mlflow, pandas as pd, numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from model_utils import TwoTowerModel, EMBED_DIM

torch.set_float32_matmul_precision("high")


# --------------------------------------------------
# caculate recall@K & upload to MLflow(use nested run)
# --------------------------------------------------
def compute_recall_at_k(model_ckpt_path, test_path="data/split/test.parquet",
                        k=50):
    print(f"\n▶️  compute_recall@{k}  (ckpt={model_ckpt_path})")

    test_df   = pd.read_parquet(test_path)
    user_enc  = pd.read_parquet("outputs/user_encoder.parquet")
    item_enc  = pd.read_parquet("outputs/item_encoder.parquet")
    test_df   = test_df.merge(user_enc,  on="user_id") \
                       .merge(item_enc, on="item_id")

    num_users = user_enc["userid"].max() + 1
    num_items = item_enc["itemid"].max() + 1
    gt_dict   = test_df.groupby("userid")["itemid"].unique().to_dict()
    test_uids = list(gt_dict.keys())

    model = TwoTowerModel.load_from_checkpoint(
        model_ckpt_path,
        num_users=num_users,
        num_items=num_items,
        embed_dim=EMBED_DIM,
    ).eval()

    user_w = model.user_emb.weight.data          # (U,D)
    item_w = model.item_emb.weight.data
    faiss.normalize_L2(item_w.cpu().numpy())

    recalls = []
    with torch.no_grad():
        for u in tqdm(test_uids, desc=f"recall@{k}"):
            scores = torch.matmul(user_w[u].unsqueeze(0), item_w.T).squeeze(0)
            topk   = torch.topk(scores, k).indices.cpu().numpy()
            hits   = np.intersect1d(gt_dict[u], topk)
            recalls.append(len(hits) / len(gt_dict[u]))

    mean_recall = float(np.mean(recalls))
    print(f"✅  Mean recall@{k}: {mean_recall:.4f}", flush=True)

    # ---------- MLflow ----------
    active = mlflow.active_run()
    if active is None:
        mlflow.set_tracking_uri("http://localhost:5500")
        mlflow.set_experiment("two_tower_recall_compute")
        mlflow.start_run(run_name=f"eval_recall@{k}")
    else:
        mlflow.start_run(run_id=active.info.run_id, nested=True)

    mlflow.log_metric(f"recall_at_{k}", mean_recall,
                      step=len(mlflow.active_run().data.metrics))
    mlflow.end_run()
    return mean_recall


# --------------------------------------------------
# for each epoch check recall@K
# --------------------------------------------------
class RecallEvalCallback(pl.Callback):
    def __init__(self, ckpt_path: str, k: int = 50):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.k         = k

    def on_validation_epoch_end(self, trainer, pl_module):
        # skip sanity check
        if trainer.sanity_checking:
            return
        trainer.save_checkpoint(self.ckpt_path)
        mrec = compute_recall_at_k(self.ckpt_path, k=self.k)
        print(f"[Epoch {trainer.current_epoch}] recall@{self.k}: "
              f"{mrec:.4f}", flush=True)


# -------------- when run seperately --------------
if __name__ == "__main__":
    compute_recall_at_k("outputs/tmp-epoch-checkpoint.ckpt", k=50)
