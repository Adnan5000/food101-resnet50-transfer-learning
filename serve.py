"""Inference service for ResNet-50 on Food-101, plus a page to try it on.

    uvicorn serve:app --port 8001

Then open http://localhost:8001 and drop in a photo.
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import transforms
from torchvision.models import resnet50

from labels import FOOD_101_CLASSES, NUM_CLASSES

WEIGHTS_PATH = Path(os.getenv("WEIGHTS_PATH", "weights/resnet50_food101.pt"))
STATIC_DIR = Path(__file__).parent / "static"

# ImageNet statistics: transfer learning keeps the pretrained normalization
# rather than recomputing it over Food-101.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224

_model: torch.nn.Module | None = None
_transform: transforms.Compose | None = None


class Prediction(BaseModel):
    predicted_class: str = Field(serialization_alias="class")
    confidence: float


class ClassifyResponse(BaseModel):
    predictions: list[Prediction]
    inference_ms: int


def load_model() -> torch.nn.Module:
    global _model, _transform

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {WEIGHTS_PATH}. Download it from the repository's "
            "Releases page, or train one with train.py."
        )

    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)

    # The checkpoint carries its own label list. Refuse to serve one whose order
    # disagrees with labels.py: every prediction would be silently mislabeled,
    # which is worse than failing to start.
    if state["classes"] != FOOD_101_CLASSES:
        raise ValueError("Checkpoint class order does not match labels.py")

    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(state["state_dict"])
    model.eval()

    _model = model
    _transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    print(f"loaded {WEIGHTS_PATH} — val top-1 {state.get('val_top1')}%")
    return model


app = FastAPI(title="Food-101 ResNet-50", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    try:
        load_model()
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        app.state.load_error = str(exc)
        print(f"model not loaded: {exc}")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "arch": "resnet50",
        "num_classes": NUM_CLASSES,
        "load_error": getattr(app.state, "load_error", None),
    }


@app.get("/", include_in_schema=False)
def demo() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/v1/classify-upload", response_model=ClassifyResponse)
async def classify_upload(file: UploadFile, top_k: int = 5) -> ClassifyResponse:
    if _model is None or _transform is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            getattr(app.state, "load_error", "Model is not loaded"),
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty upload")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - a bad upload is the client's problem
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"Could not read image: {exc}"
        ) from exc

    started = time.perf_counter()
    batch = _transform(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(_model(batch), dim=1)[0]
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    k = max(1, min(top_k, 10))
    scores, indices = probabilities.topk(k)
    return ClassifyResponse(
        predictions=[
            Prediction(predicted_class=FOOD_101_CLASSES[i], confidence=round(float(s), 4))
            for s, i in zip(scores.tolist(), indices.tolist(), strict=True)
        ],
        inference_ms=elapsed_ms,
    )
