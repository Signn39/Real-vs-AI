from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a supported image."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def prepare_image(file: BinaryIO, max_bytes: int = MAX_UPLOAD_BYTES) -> np.ndarray:
    """Decode and normalize an upload into a model-ready RGB array."""
    raw = file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ImageValidationError(
            "FILE_TOO_LARGE",
            "Please upload an image no larger than 10 MB.",
            413,
        )
    if not raw:
        raise ImageValidationError("MISSING_IMAGE", "Please select an image.")

    try:
        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    "INVALID_FILE_TYPE",
                    "Please upload a JPG, PNG, or WEBP image.",
                    415,
                )
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            normalized = normalized.resize((224, 224), Image.Resampling.LANCZOS)
            return np.asarray(normalized, dtype=np.float32)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError) as error:
        raise ImageValidationError(
            "INVALID_IMAGE",
            "The uploaded file is not a valid image.",
        ) from error
