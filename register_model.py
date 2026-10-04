import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "iris-dummy"
MODEL_NAME = "iris-classifier"

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)[0]
print(f"En iyi run: {best_run.info.run_name} accuracy={best_run.data.metrics['accuracy']:.3f}")

model_version = mlflow.register_model(f"runs:/{best_run.info.run_id}/model", MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "champion", model_version.version)

print(f"{MODEL_NAME} v{model_version.version} -> @champion")
