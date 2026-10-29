import pytest

from image_classifier.config import Settings


def test_settings_use_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IMAGE_CLASSIFIER_API_TOKEN", raising=False)
    monkeypatch.delenv("IMAGE_CLASSIFIER_MAX_UPLOAD_BYTES", raising=False)
    monkeypatch.delenv("IMAGE_CLASSIFIER_MAX_IMAGE_PIXELS", raising=False)
    settings = Settings.from_environment()
    assert settings.api_token is None
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.max_image_pixels == 40_000_000


@pytest.mark.parametrize("value", ["0", "-10", "lots"])
def test_settings_reject_invalid_limits(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("IMAGE_CLASSIFIER_MAX_UPLOAD_BYTES", value)
    with pytest.raises(ValueError, match="positive integer"):
        Settings.from_environment()

