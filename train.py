import sys

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-dummy")

C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = LogisticRegression(C=C, max_iter=200).fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("C", C)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3))

    print(f"C={C} accuracy={acc:.3f}")
