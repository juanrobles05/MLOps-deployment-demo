from pathlib import Path

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos y modelos
DATA_PATH = ROOT_DIR / "data" / "Telco-Customer-Churn.csv"
RAW_PATH = DATA_PATH
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
METADATA_PATH = ROOT_DIR / "models" / "metadata.json"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

# Semilla fija para reproducibilidad
SEED = 42

# Nombre de la variable objetivo
TARGET = "Churn"

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
