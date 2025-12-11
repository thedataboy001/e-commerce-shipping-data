import json
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import joblib
import mlflow
import mlflow.sklearn


from src.utils_and_constants import RANDOM_STATE, ARTIFACTS_DIR

def train_random_forest(X_train, y_train, X_test, y_test):

    model = RandomForestClassifier(n_estimators= 600, 
                                   min_samples_split = 2, min_samples_leaf = 3, 
                                   max_features = None, max_depth = None, 
                                   class_weight = 'balanced_subsample', 
                                   bootstrap = False, 
                                   random_state= RANDOM_STATE)
    
    mlflow.set_experiment("ecommerce_shipping_rf")

    with mlflow.start_run(run_name="random_forest_v1"):


        # Train model     
        model.fit(X_train, y_train)

        # Predict Model
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average= 'binary')

        # log params

        mlflow.log_params(model.get_params())

        # Log metrics
            
        mlflow.log_metric("accuracy" , accuracy)
        mlflow.log_metric("precision" , precision)
        mlflow.log_metric("recall" , recall)
        mlflow.log_metric("f1_score" , f1)

        # Log model

        mlflow.sklearn.log_model(sk_model=model, artifact_path="model",   # how it appears in MLflow UI
                                 registered_model_name=None 
        )

        # ---- Save model for API inference (stable path) ----
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = ARTIFACTS_DIR / "model.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Saved model for inference to {model_path}")

    return model


def plot_confusion_metrix(model, X_test, y_test):
    _ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
    plt.savefig(f"confusion_metrix.png")