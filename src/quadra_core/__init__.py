"""Read a supermarket till receipt from a photo.

Tesseract does the OCR, then geometry and arithmetic do the rest. The same
receipt always parses the same way, and when it gets one wrong you can see why.

    from quadra_core import parse_image, parse_tsv

    document, status = parse_image(Path("receipt.jpg"), profile_id="esselunga")

`status` is "ok", "partial" or "failed". It never raises just because a receipt
was hard to read: you get a document back either way, with the problems listed
in `validation.warnings`.

Everything shop-specific lives in a TOML profile rather than in this code, so
adding a shop means writing a profile.
"""

from __future__ import annotations

from pathlib import Path

from quadra_core.parse import parse_image, parse_tsv, read_input
from quadra_core.pipeline.document import no_review
from quadra_core.profiles.loader import Profile, ProfileError, available, load, select

# The JSON Schema every document validates against. Shipped with the package so
# callers can check output without keeping their own copy in sync.
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
