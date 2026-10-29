"""Command-line prediction interface."""

import argparse
import json
from pathlib import Path

from image_classifier.config import Settings
from image_classifier.image_io import InvalidImageError, decode_image
from image_classifier.predictor import ImagePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an image with a pretrained ResNet-50 model.")
    parser.add_argument("image", type=Path, help="Path to a PNG, JPEG, or WebP image")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to return (1-1000)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 1000:
        parser.error("--top-k must be between 1 and 1000")
    try:
        settings = Settings.from_environment()
        if args.image.stat().st_size > settings.max_upload_bytes:
            raise ValueError("The image file exceeds the configured size limit.")
        data = args.image.read_bytes()
        image = decode_image(data, max_pixels=settings.max_image_pixels)
        predictions = ImagePredictor().predict(image, top_k=args.top_k)
    except (OSError, InvalidImageError, ValueError) as exc:
        parser.error(str(exc))
    result = [{"label": item.label, "confidence": item.confidence} for item in predictions]
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for index, item in enumerate(result, start=1):
            print(f"{index}. {item['label']}: {item['confidence']:.2%}")


if __name__ == "__main__":
    main()

