FROM python:3.10-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/__init__.py src/api.py ./src/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]