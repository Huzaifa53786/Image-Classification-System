"""FastAPI application with a small browser upload page and JSON API."""

from contextlib import asynccontextmanager
import hmac
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from image_classifier.config import Settings
from image_classifier.image_io import InvalidImageError, decode_image
from image_classifier.predictor import ImagePredictor


class PredictionItem(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    model: str
    predictions: list[PredictionItem]


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Classifier</title><style>
body{font:16px system-ui,sans-serif;max-width:720px;margin:10vh auto;padding:0 20px;color:#172033;background:#f5f7fb}
main{background:white;border:1px solid #dce2ed;border-radius:16px;padding:28px;box-shadow:0 12px 36px #17203312}
button{background:#2459d3;color:white;border:0;border-radius:8px;padding:11px 18px;font-weight:650;cursor:pointer}
input{margin:18px 0}li{margin:10px 0}.muted{color:#536174}#status{white-space:pre-wrap}
</style></head><body><main><h1>Image Classifier</h1>
<p class="muted">Upload a PNG, JPEG, or WebP image. The model returns its top ImageNet-1K matches.</p>
<form id="form"><input name="file" type="file" accept="image/png,image/jpeg,image/webp" required>
<label>Optional API token <input name="api_token" type="password" autocomplete="off"></label><br>
<label>Results <select name="top_k"><option>5</option><option>1</option><option>3</option><option>10</option></select></label>
<button>Classify image</button></form><p id="status" role="status"></p><ol id="results"></ol></main>
<script>const form=document.querySelector('#form'),status=document.querySelector('#status'),results=document.querySelector('#results');
form.addEventListener('submit',async e=>{e.preventDefault();status.textContent='Classifying…';results.replaceChildren();
const data=new FormData(form),k=data.get('top_k'),token=data.get('api_token');data.delete('top_k');data.delete('api_token');const headers=token?{Authorization:'Bearer '+token}:{};try{const r=await fetch('/predict?top_k='+k,{method:'POST',body:data,headers});const body=await r.json();
if(!r.ok)throw Error(body.detail||'Request failed');for(const p of body.predictions){const li=document.createElement('li');li.textContent=p.label+' — '+(p.confidence*100).toFixed(2)+'%';results.append(li)}status.textContent='Model: '+body.model;
}catch(err){status.textContent=err.message}});
</script></body></html>"""


def create_app(
    *, predictor: ImagePredictor | None = None, settings: Settings | None = None
) -> FastAPI:
    """Create an app; accepting injected dependencies keeps API behavior testable."""
    runtime_settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor or ImagePredictor()
        yield

    app = FastAPI(
        title="Image Classification API",
        description="Classify PNG, JPEG, and WebP images using pretrained ImageNet-1K weights.",
        version="1.0.0",
        lifespan=lifespan,
    )

    async def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = runtime_settings.api_token
        if expected is None:
            return
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="A valid bearer token is required.")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def home() -> str:
        return PAGE

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readiness() -> dict[str, str]:
        if not hasattr(app.state, "predictor"):
            raise HTTPException(status_code=503, detail="Model is loading.")
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(authorize)])
    async def predict(
        file: Annotated[UploadFile, File(description="PNG, JPEG, or WebP image")],
        top_k: Annotated[int, Query(ge=1, le=1000)] = 5,
    ) -> PredictionResponse:
        data = await file.read(runtime_settings.max_upload_bytes + 1)
        await file.close()
        if len(data) > runtime_settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="The uploaded file exceeds the size limit.")
        try:
            image = decode_image(data, max_pixels=runtime_settings.max_image_pixels)
        except InvalidImageError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        try:
            predictions = app.state.predictor.predict(image, top_k=top_k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionResponse(
            model="resnet50-imagenet1k-v2",
            predictions=[
                PredictionItem(label=item.label, confidence=item.confidence) for item in predictions
            ],
        )

    return app


app = create_app()

