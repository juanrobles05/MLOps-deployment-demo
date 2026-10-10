"""Entrena y registra el mejor modelo de churn."""
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from data import load_data


def create_pipeline(model):
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), config.NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main():
    data = load_data()
    X = data[config.FEATURES]
    y = data[config.TARGET]
    X[config.NUMERIC_FEATURES] = X[config.NUMERIC_FEATURES].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.SEED
    )
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(
        config.PROCESSED_DIR / "train.csv", index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(
        config.PROCESSED_DIR / "test.csv", index=False
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=config.SEED,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    results = []
    mlflow.set_experiment("Telco_Churn_Experiment")

    for name, model in models.items():
        pipeline = create_pipeline(model)
        with mlflow.start_run(run_name=name) as run:
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
            pipeline.fit(X_train, y_train)
            probability = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probability)
            f1 = f1_score(y_test, probability >= 0.5, zero_division=0)

            mlflow.log_param("model", name)
            mlflow.log_metric("cv_auc", scores.mean())
            mlflow.log_metric("test_auc", auc)
            mlflow.log_metric("test_f1", f1)
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1]),
                serialization_format="skops",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            results.append((scores.mean(), run.info.run_id, name, pipeline))
            print(f"{name}: CV AUC={scores.mean():.4f} | F1={f1:.4f}")

    best_auc, best_run_id, best_name, best_pipeline = max(results, key=lambda item: item[0])
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, config.MODEL_PATH)
    config.METADATA_PATH.write_text(json.dumps({
        "model": best_name,
        "mlflow_run_id": best_run_id,
        "cv_auc": round(best_auc, 4),
    }))
    print(f"Mejor modelo: {best_name} | CV AUC={best_auc:.4f}")
    print(f"Guardado en: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
