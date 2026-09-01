# Production image for the churn-scoring FastAPI service.
# Build:  docker build -t churn-service .
# Run:    docker run -p 8000:8000 churn-service

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer) so code changes don't bust the
# dependency-install cache on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the trained model artifact.
# In a real pipeline, the model artifact would be pulled from a model
# registry (MLflow, S3) at build or startup time rather than baked into the
# image -- baking it in here for simplicity of a single deployable image.
COPY src/ src/
COPY api/ api/
COPY config/ config/
COPY saved_models/ saved_models/

# Run as a non-root user -- standard production hardening.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
