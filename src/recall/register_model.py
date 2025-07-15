import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5500")

run_id = "ca1dce4538734bd2a54432fbb32953e9"
artifact_path = "two_tower_checkpoint.pt"
model_name = "TwoTowerRecSys"

model_uri = f"runs:/{run_id}/{artifact_path}"

client = MlflowClient()

result = client.create_registered_model(model_name)
model_version = client.create_model_version(model_name, model_uri, run_id)

print("Model registered successfully!")
print("Name:", model_name)
print("Version:", model_version.version)
