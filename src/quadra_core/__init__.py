"""Read a supermarket till receipt from a photo.

`parse_image(path)` returns a Result whose `status` is "ok", "review" or "failed".
With "review", `document["review"]` says what a person should check, field by field.
"""

from __future__ import annotations

from pathlib import Path

from quadra_core.parse import Result, parse_image, parse_tsv
from quadra_core.pipeline.ocr import OcrError
from quadra_core.profiles.loader import ProfileError

# The JSON Schema every document validates against.
SCHEMA_PATH = Path(__file__).parent / "schema" / "receipt-2.0.0.schema.json"

__all__ = [
    "SCHEMA_PATH",
    "OcrError",
    "ProfileError",
    "Result",
    "parse_image",
    "parse_tsv",
]
