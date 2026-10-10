"""Simula dos escenarios de drift y decide qué hacer en cada uno.

Escenario A (data drift):    sube MonthlyCharges; las etiquetas reales NO cambian.
Escenario B (concept drift): además, parte de los clientes leales empieza a irse.

Importante: los datos de "producción" son SIMULADOS a partir del conjunto de test.

Uso:  python -m src.monitor [--shift 1.3] [--flip-frac 0.3]
"""
import argparse
import json
import re

import joblib
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from . import config

AUC_DROP_ALERT = 0.05  # regla de decisión: caída de AUC que justifica reentrenar


def drifted_columns(snapshot) -> dict:
    """Extrae de Evidently las columnas cuyo puntaje de drift supera su umbral."""
    found = {}
    for metric in snapshot.dict()["metrics"]:
        m = re.match(r"ValueDrift\(column=(.+?),method=.+?,threshold=([\d.]+)\)", metric["metric_name"])
        if m and metric["value"] > float(m.group(2)):
            found[m.group(1)] = round(float(metric["value"]), 3)
    return found


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = proba >= 0.5
    return {
        "churn_rate": round(float(y.mean()), 3),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "f1": round(f1_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
    }


def main(shift: float, flip_frac: float) -> None:
    model = joblib.load(config.MODEL_PATH)
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    test = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    X_ref, X_test, y_test = train[config.FEATURES], test[config.FEATURES], test[config.TARGET]

    # --- Construcción de los escenarios simulados -----------------------------
    X_a = X_test.copy()
    X_a["MonthlyCharges"] *= shift                       # A: cambia la entrada
    y_a = y_test.copy()                                  #    etiquetas intactas

    rng = np.random.default_rng(config.SEED)
    y_b = y_test.copy()                                  # B: cambia la relación X -> y
    loyal_stayers = (X_a["tenure"] >= 36) & (y_b == 0)
    y_b[loyal_stayers & (rng.random(len(y_b)) < flip_frac)] = 1

    # --- Detección de data drift con Evidently (solo mira las entradas) --------
    definition = DataDefinition(
        numerical_columns=config.NUMERIC_FEATURES,
        categorical_columns=config.CATEGORICAL_FEATURES,
    )
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(X_a, data_definition=definition),
        Dataset.from_pandas(X_ref, data_definition=definition),
    )
    config.REPORTS_DIR.mkdir(exist_ok=True)
    snapshot.save_html(str(config.REPORTS_DIR / "drift_report.html"))
    drifted = drifted_columns(snapshot)

    # --- Desempeño (requiere etiquetas, que en producción llegan tarde) --------
    ref_metrics = evaluate(model, X_test, y_test)
    scenarios = {
        "A: data drift": evaluate(model, X_a, y_a),
        "B: data + concept drift": evaluate(model, X_a, y_b),
    }

    print(f"\nColumnas con data drift (Evidently): {drifted or 'ninguna'}")
    print(f"\n{'Escenario':26s} {'AUC':>7s} {'F1':>7s} {'Recall':>7s} {'Churn real':>11s}  Decisión")
    print("-" * 100)
    print(f"{'Referencia (test)':26s} {ref_metrics['roc_auc']:7.4f} {ref_metrics['f1']:7.4f} "
          f"{ref_metrics['recall']:7.4f} {ref_metrics['churn_rate']:11.3f}  -")
    summary = {"drifted_columns": drifted, "reference": ref_metrics, "scenarios": {}}
    for name, m in scenarios.items():
        drop = ref_metrics["roc_auc"] - m["roc_auc"]
        if drop > AUC_DROP_ALERT:
            decision = "REENTRENAR: el desempeño cayó (validar y desplegar con canary)"
        elif drifted:
            decision = "INVESTIGAR: ¿cambio legítimo o error de datos? Aún no reentrenar"
        else:
            decision = "OK"
        print(f"{name:26s} {m['roc_auc']:7.4f} {m['f1']:7.4f} {m['recall']:7.4f} "
              f"{m['churn_rate']:11.3f}  {decision}")
        summary["scenarios"][name] = {**m, "auc_drop": round(drop, 4), "decision": decision}

    (config.REPORTS_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nReporte HTML: {(config.REPORTS_DIR / 'drift_report.html').relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shift", type=float, default=1.3, help="factor sobre MonthlyCharges")
    parser.add_argument("--flip-frac", type=float, default=0.3,
                        help="fracción de clientes leales que abandona en el escenario B")
    args = parser.parse_args()
    main(args.shift, args.flip_frac)
