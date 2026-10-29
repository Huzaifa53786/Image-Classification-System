"""ImageNet classification backed by TorchVision's pretrained ResNet-50."""

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class ImagePredictor:
    """Loads pretrained weights once and serves top-k ImageNet predictions."""

    def __init__(self) -> None:
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=self.weights)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    @torch.inference_mode()
    def predict(self, image: Image.Image, *, top_k: int = 5) -> list[Prediction]:
        if not 1 <= top_k <= len(self.categories):
            raise ValueError(f"top_k must be between 1 and {len(self.categories)}")
        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        probabilities = self.model(batch).softmax(dim=1)[0]
        scores, indexes = probabilities.topk(top_k)
        return [
            Prediction(label=self.categories[index], confidence=float(score))
            for score, index in zip(scores.tolist(), indexes.tolist(), strict=True)
        ]

