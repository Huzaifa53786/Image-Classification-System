from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from image_classifier.api import create_app
from image_classifier.config import Settings


class FakePredictor:
    def predict(self, image: Image.Image, *, top_k: int = 5):
        assert image.mode == "RGB"
        return [type("Result", (), {"label": "tabby cat", "confidence": 0.91})()][:top_k]


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (4, 4), color="red").save(stream, format="PNG")
    return stream.getvalue()


def test_predict_endpoint_returns_ranked_results() -> None:
    app = create_app(predictor=FakePredictor(), settings=Settings(None, 1024, 100))
    with TestClient(app) as client:
        response = client.post("/predict", files={"file": ("cat.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["predictions"][0] == {"label": "tabby cat", "confidence": 0.91}


def test_predict_endpoint_rejects_oversized_upload() -> None:
    app = create_app(predictor=FakePredictor(), settings=Settings(None, 10, 100))
    with TestClient(app) as client:
        response = client.post("/predict", files={"file": ("large.png", png_bytes(), "image/png")})
    assert response.status_code == 413


def test_predict_endpoint_can_require_bearer_token() -> None:
    app = create_app(predictor=FakePredictor(), settings=Settings("secret", 1024, 100))
    with TestClient(app) as client:
        denied = client.post("/predict", files={"file": ("cat.png", png_bytes(), "image/png")})
        allowed = client.post(
            "/predict",
            files={"file": ("cat.png", png_bytes(), "image/png")},
            headers={"Authorization": "Bearer secret"},
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200

