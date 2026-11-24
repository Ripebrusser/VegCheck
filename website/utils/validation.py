"""Upload validation and safe decoding.

Streamlit accepts whatever the user selects, so every assumption about the file
has to be checked here: extension, size, decodability, channel count and
dimensions. A failure raises :class:`ValidationError` with a message meant for a
human, and the UI renders it instead of a traceback.
"""

from __future__ import annotations

import numpy as np

from website.utils.paths import (
    ALLOWED_EXTENSIONS,
    MAX_SIDE,
    MAX_UPLOAD_BYTES,
    MIN_SIDE,
)


class ValidationError(ValueError):
    """A rejected upload, with a message safe to show to the user."""


def extension_of(filename: str) -> str:
    name = (filename or "").strip().lower()
    dot = name.rfind(".")
    return name[dot:] if dot >= 0 else ""


def validate_upload(filename: str, data: bytes | None) -> None:
    """Check everything that can be checked without decoding."""
    if data is None or len(data) == 0:
        raise ValidationError("The uploaded file is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        mb = len(data) / 1024 ** 2
        raise ValidationError(
            f"File is {mb:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1024 ** 2:.0f} MB.")

    ext = extension_of(filename)
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise ValidationError(
            f"Unsupported file type '{ext or filename}'. Accepted types: {allowed}.")

    # Magic-number sanity: an extension is a claim, not evidence.
    if not _looks_like_image(data):
        raise ValidationError(
            "The file content is not a readable JPEG or PNG image, "
            "regardless of its extension.")


def _looks_like_image(data: bytes) -> bool:
    if data[:3] == b"\xff\xd8\xff":                      # JPEG SOI
        return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":                 # PNG signature
        return True
    return False


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes to an RGB uint8 array, or raise :class:`ValidationError`."""
    import cv2

    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValidationError(
            "The image could not be decoded. It may be corrupted or truncated.")

    height, width = bgr.shape[:2]
    if height < MIN_SIDE or width < MIN_SIDE:
        raise ValidationError(
            f"Image is {width}×{height}; the minimum size is {MIN_SIDE}×{MIN_SIDE}.")
    if height > MAX_SIDE or width > MAX_SIDE:
        raise ValidationError(
            f"Image is {width}×{height}; the maximum size is {MAX_SIDE}×{MAX_SIDE}.")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValidationError("Image could not be converted to three RGB channels.")
    return np.ascontiguousarray(rgb)


def describe_upload(filename: str, data: bytes, rgb: np.ndarray) -> dict:
    """Facts about accepted upload, for display."""
    height, width = rgb.shape[:2]
    return {
        "filename": filename,
        "bytes": len(data),
        "size_mb": len(data) / 1024 ** 2,
        "width": int(width),
        "height": int(height),
        "megapixels": (width * height) / 1e6,
        "extension": extension_of(filename),
    }
