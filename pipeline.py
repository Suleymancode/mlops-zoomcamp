import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from prefect import flow, task
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "iris-dummy"
MODEL_NAME = "iris-classifier"

mlflow.set_tracking_uri("http://127.0.0.1:5000")


@task
def load_data():
    X, y = load_iris(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@task(retries=2, retry_delay_seconds=5)
def train_and_log(C, X_train, X_test, y_train, y_test):
    with mlflow.start_run() as run:
        model = LogisticRegression(C=C, max_iter=200).fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3))

    print(f"C={C} accuracy={acc:.3f} run_id={run.info.run_id}")
    return run.info.run_id, acc


@task
def register_if_better(run_id, acc):
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_acc = client.get_run(champion.run_id).data.metrics["accuracy"]
    except MlflowException:
        champion_acc = -1.0

    if acc <= champion_acc:
        print(f"Champion daha iyi veya eşit ({champion_acc:.3f} >= {acc:.3f}), kayıt yapılmadı.")
        return None

    model_version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", model_version.version)
    print(f"Yeni champion: {MODEL_NAME} v{model_version.version} ({acc:.3f} > {champion_acc:.3f})")
    return model_version.version


@flow(log_prints=True)
def training_pipeline(c_values: list[float] = [0.01, 0.1, 1.0]):
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()
    results = [train_and_log(C, X_train, X_test, y_train, y_test) for C in c_values]

    best_run_id, best_acc = max(results, key=lambda r: r[1])
    register_if_better(best_run_id, best_acc)


if __name__ == "__main__":
    training_pipeline.serve(name="iris-training", cron="0 3 * * *")
