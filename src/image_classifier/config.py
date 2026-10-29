"""Runtime configuration loaded from environment variables."""

from dataclasses import dataclass
import os


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Settings:
    """Validated service settings."""

    api_token: str | None
    max_upload_bytes: int
    max_image_pixels: int

    @classmethod
    def from_environment(cls) -> "Settings":
        token = os.getenv("IMAGE_CLASSIFIER_API_TOKEN") or None
        return cls(
            api_token=token,
            max_upload_bytes=_positive_int("IMAGE_CLASSIFIER_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
            max_image_pixels=_positive_int("IMAGE_CLASSIFIER_MAX_IMAGE_PIXELS", 40_000_000),
        )

