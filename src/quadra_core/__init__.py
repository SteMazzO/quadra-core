"""Read a supermarket till receipt from a photo.

    document, status = parse_image(Path("receipt.jpg"), profile_id="esselunga")

`status` is "ok", "partial" or "failed"; problems are in `validation.warnings`.
"""

from __future__ import annotations

from pathlib import Path

from quadra_core.parse import parse_image, parse_tsv, read_input
from quadra_core.pipeline.document import no_review
from quadra_core.profiles.loader import Profile, ProfileError, available, load, select

# The JSON Schema every document validates against.
SCHEMA_PATH = Path(__file__).parent / "schema" / "receipt-1.0.0.schema.json"

__all__ = [
    "SCHEMA_PATH",
    "Profile",
    "ProfileError",
    "available",
    "load",
    "no_review",
    "parse_image",
    "parse_tsv",
    "read_input",
    "select",
]
