FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/models

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser && mkdir -p /models && chown appuser:appuser /models

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    python -m pip install --no-deps . && \
    python -m pip install fastapi 'uvicorn[standard]' pillow python-multipart

USER appuser
EXPOSE 8000
CMD ["uvicorn", "image_classifier.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

