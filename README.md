# Deep Learning Image Classification System

A production-oriented image classification project using **Python, PyTorch, TorchVision, and ResNet-50**. Upload an image in a browser, call the HTTP API, or use the command line. The pretrained model returns its top matches and confidence scores.

> **Supported classes:** this project uses ImageNet-1K pretrained weights and predicts among 1,000 known categories. It does not recognize every possible object, text, scene, or custom class. Confidence scores are model probabilities, not calibrated guarantees.

## Features

- Browser upload page and JSON API with health and readiness endpoints.
- CLI for local batch-by-batch image prediction.
- PNG, JPEG, and WebP decoding with file-size and image-pixel limits.
- Optional bearer-token authentication for the prediction endpoint.
- CPU inference by default; GPU use is available when a compatible PyTorch build and device are installed.
- Docker and Docker Compose support, non-root container user, persistent pretrained-weight cache.
- GitHub Actions lint and test workflow.

## Run locally

Python 3.10 or later is required. Install the CPU build of PyTorch and then this package:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -e ".[dev]"
```

For CUDA or another accelerator, select the matching PyTorch and TorchVision build from the [official PyTorch install selector](https://pytorch.org/get-started/locally/) instead of the CPU index.

Start the web app:

```powershell
uvicorn image_classifier.api:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and upload an image. Interactive API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The first prediction downloads the pretrained weights (about 98 MB); later runs reuse the Torch cache.

## Use the API

```bash
curl -X POST "http://127.0.0.1:8000/predict?top_k=5" \
  -F "file=@./image.jpg"
```

Example response:

```json
{
  "model": "resnet50-imagenet1k-v2",
  "predictions": [
    { "label": "tabby cat", "confidence": 0.71 }
  ]
}
```

`top_k` must be from 1 to 1,000. The API verifies image content instead of trusting the filename or browser MIME type. Invalid formats return HTTP 415, oversized files return HTTP 413, and image dimensions above the configured pixel cap are rejected.

To protect an internet-facing service, set a long random token in `.env` (copy `.env.example`) and send it as a bearer token:

```bash
curl -X POST "https://your-host/predict?top_k=5" \
  -H "Authorization: Bearer YOUR_SECRET_TOKEN" \
  -F "file=@./image.jpg"
```

The browser page includes an optional API token field. Tokens are sent only in the request authorization header; use HTTPS when accessing the service remotely. You can also call the API with a bearer header or place the service behind an authenticated gateway. Set TLS, rate limits, and request timeouts at the reverse proxy or hosting platform.

## Use the CLI

```bash
image-classifier ./image.jpg --top-k 5
image-classifier ./image.jpg --top-k 3 --json
```

The first run downloads the same pretrained weights. Use `TORCH_HOME` to select a persistent cache directory.

## Docker

```bash
docker compose up --build
```

Open http://localhost:8000. Docker Compose persists model weights in a named volume. Configure optional limits and API authentication in a local `.env` file. For a single container:

```bash
docker build -t image-classifier .
docker run --rm -p 8000:8000 -v image-model-cache:/models image-classifier
```

The first container start needs network access to download model weights. CPU inference has higher latency than a GPU. Run one Uvicorn worker per container to avoid loading a separate large model copy in each worker; scale with additional containers if needed.

## Tests and quality checks

```bash
python -m pytest
ruff check .
```

The tests cover image decoding and safety limits, environment configuration, API prediction responses, upload rejection, and optional authentication. GitHub Actions runs lint and tests on Python 3.10 and 3.12.

## GitHub deployment

Create a new GitHub repository, then from this project folder:

```bash
git init
git add .
git commit -m "Build pretrained image classification service"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

The Actions workflow runs automatically after the push. To run the service, deploy the repository to a Python or Docker hosting platform. GitHub Pages only hosts static sites and cannot run the PyTorch API. Configure the hosting provider to expose port 8000 and set the environment variables described in `.env.example`. Keep secrets in the provider's secret manager; do not commit `.env`.

## Model and responsible use

The model and image preprocessing are supplied by TorchVision's [ResNet-50 ImageNet weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html). Categories and preprocessing are taken directly from the selected weights. Predictions can be wrong, reflect ImageNet label limits and training biases, and should not be used alone for consequential decisions. Uploaded images are processed in memory and are not intentionally written to disk by the application.

This project provides inference, not model training. To classify domain-specific categories beyond ImageNet-1K, collect and label an appropriate dataset and fine-tune/evaluate a model for that use case.

## License

Project source is licensed under the MIT License. The pretrained model weights retain their upstream terms; review them before redistribution.

