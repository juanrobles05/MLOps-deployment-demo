# MLOps Deployment Demo

Proyecto de ejemplo para mostrar un flujo sencillo de MLOps con el dataset
Telco Customer Churn:

```text
dataset -> entrenamiento -> modelo local -> API -> monitoreo
```

El proyecto compara modelos de clasificación, registra los experimentos en
MLflow, guarda el mejor modelo y lo expone mediante una API REST.

## Requisitos

- Python 3.10 o superior
- Docker (opcional)

## Instalación

Desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Entrenamiento

El dataset predeterminado es `data/Telco-Customer-Churn.csv`.

```powershell
python -m src.train
```

El entrenamiento:

- limpia el dataset;
- compara regresión logística y Random Forest;
- selecciona el mejor modelo por AUC;
- registra los experimentos en MLflow;
- guarda el modelo en `models/model.joblib`;
- genera las particiones usadas por el monitoreo en `data/processed/`.

Para usar otra versión del dataset, cambia la ruta en `src/config.py` o
actualiza `DATA_PATH`.

## API

Primero ejecuta el entrenamiento y después inicia el servidor:

```powershell
uvicorn src.api:app --reload --port 8000
```

Documentación interactiva:

<http://127.0.0.1:8000/docs>

Endpoints principales:

- `GET /health`
- `POST /predict`

La API carga el modelo guardado en `models/model.joblib`.

## Monitoreo

Ejecuta el entrenamiento al menos una vez para crear las particiones de datos.
Después genera el reporte de drift:

```powershell
python -m src.monitor
```

Los resultados se guardan en:

```text
reports/drift_report.html
reports/drift_summary.json
```

El monitor simula data drift modificando `MonthlyCharges` y concept drift
modificando algunas etiquetas.

## Docker

El contenedor sirve la API y utiliza el modelo previamente entrenado. Por eso
debes ejecutar el entrenamiento antes de construir la imagen:

```powershell
python -m src.train
docker build -t mlops-demo .
docker run --rm -p 8000:8000 mlops-demo
```

Después abre:

<http://127.0.0.1:8000/docs>

La imagen incluye únicamente `src/` y `models/`. El dataset, los notebooks,
los reportes y los archivos locales de MLflow se usan fuera del contenedor.

## Notebook

El análisis exploratorio está disponible en:

```text
notebooks/eda.ipynb
```

## Estructura

```text
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   └── model.joblib          # generado por el entrenamiento
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── api.py
│   ├── config.py
│   ├── data.py
│   ├── monitor.py
│   └── train.py
├── Dockerfile
├── requirements.txt
└── Makefile
```

Los modelos, reportes, datos procesados y archivos de MLflow generados
localmente están excluidos de Git mediante `.gitignore`.
