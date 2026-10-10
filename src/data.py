"""Descarga verificada y limpieza del dataset Telco Customer Churn."""
import pandas as pd

if __package__:
    from . import config
else:
    import config


def load_data() -> pd.DataFrame:
    """Devuelve el dataset limpio: features + columna objetivo binaria (0/1)."""
    df = pd.read_csv(config.RAW_PATH)
    # TotalCharges viene como texto y tiene 11 espacios vacíos (clientes con
    # tenure = 0, que aún no han recibido factura): se convierten a 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    return df[config.FEATURES + [config.TARGET]]


def load_and_clean_data() -> pd.DataFrame:
    """Alias usado por el script de monitoreo."""
    return load_data()