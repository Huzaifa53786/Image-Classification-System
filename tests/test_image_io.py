from io import BytesIO

import pytest
from PIL import Image

from image_classifier.image_io import InvalidImageError, decode_image


def make_png(size: tuple[int, int] = (4, 3)) -> bytes:
    stream = BytesIO()
    Image.new("RGBA", size, color=(255, 0, 0, 128)).save(stream, format="PNG")
    return stream.getvalue()


def test_decode_image_converts_supported_image_to_rgb() -> None:
    image = decode_image(make_png(), max_pixels=100)
    assert image.mode == "RGB"
    assert image.size == (4, 3)


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_decode_image_rejects_empty_or_invalid_bytes(data: bytes) -> None:
    with pytest.raises(InvalidImageError):
        decode_image(data, max_pixels=100)


def test_decode_image_rejects_images_over_pixel_limit() -> None:
    with pytest.raises(InvalidImageError, match="pixel limit"):
        decode_image(make_png((11, 10)), max_pixels=100)


def test_decode_image_rejects_unsupported_format() -> None:
    stream = BytesIO()
    Image.new("RGB", (2, 2)).save(stream, format="BMP")
    with pytest.raises(InvalidImageError, match="PNG, JPEG, or WebP"):
        decode_image(stream.getvalue(), max_pixels=100)

