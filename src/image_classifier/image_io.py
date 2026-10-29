"""Safe decoding and validation for user-supplied images."""

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a supported, valid image."""


def decode_image(data: bytes, *, max_pixels: int) -> Image.Image:
    """Decode PNG, JPEG, or WebP bytes as RGB after validating dimensions and content."""
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    try:
        with Image.open(BytesIO(data)) as probe:
            if probe.format not in {"PNG", "JPEG", "WEBP"}:
                raise InvalidImageError("Use a PNG, JPEG, or WebP image.")
            width, height = probe.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise InvalidImageError(f"Image dimensions exceed the {max_pixels:,}-pixel limit.")
            probe.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("The upload is not a valid supported image.") from exc

