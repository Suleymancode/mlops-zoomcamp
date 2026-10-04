import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel

MODEL_NAME = "iris-classifier"
MODEL_ALIAS = "champion"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

state = {}


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model, her istekte değil, servis açılırken bir kez yüklenir.
    state["model"] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    state["version"] = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    print(f"Yüklendi: {MODEL_NAME}@{MODEL_ALIAS} (v{state['version']})")
    yield
    state.clear()


app = FastAPI(title="Iris Prediction Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": state.get("version")}


@app.post("/predict")
def predict(features: IrisFeatures):
    # Model signature eğitimdeki kolon adlarını bekliyor.
    df = pd.DataFrame([{
        "sepal length (cm)": features.sepal_length,
        "sepal width (cm)": features.sepal_width,
        "petal length (cm)": features.petal_length,
        "petal width (cm)": features.petal_width,
    }])
    pred = int(state["model"].predict(df)[0])
    return {"class_id": pred, "class_name": CLASS_NAMES[pred], "model_version": state["version"]}
