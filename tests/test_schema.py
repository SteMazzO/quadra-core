"""Every parser output must validate against the published JSON Schema.

The schema is the contract other tools read the data through, so it is checked
against real fixture output rather than a hand-written example that can drift.
"""

from __future__ import annotations

import json

import pytest

jsonschema = pytest.importorskip("jsonschema")

from quadra_core import SCHEMA_PATH, parse_tsv  # noqa: E402
from quadra_core.testdata import OCR as FIXTURES  # noqa: E402

CASES = [
    ("esselunga_photo_payments", "esselunga"),
    ("esselunga_photo_modifier", "esselunga"),
    ("esselunga_a", "esselunga"),
    ("esselunga_b", "esselunga"),
    ("esselunga_b_faded", "esselunga"),
    ("synthetic_clean", "synthetic"),
    ("synthetic_faded", "synthetic"),
    ("synthetic_unbalanced", "synthetic"),
]


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text())


def test_schema_is_itself_valid(schema):
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("name,profile", CASES)
def test_fixture_output_validates(schema, name, profile):
    document, _ = parse_tsv(
        (FIXTURES / f"{name}.tsv").read_text(),
        profile_id=profile,
        receipt_id=name,
    )
    jsonschema.Draft202012Validator(schema).validate(document)


def test_schema_rejects_float_money(schema):
    """Money must be integer minor units; a float would silently drift a cent."""
    document, _ = parse_tsv(
        (FIXTURES / "esselunga_b.tsv").read_text(),
        profile_id="esselunga",
        receipt_id="x",
    )
    document["line_items"][0]["line_total_minor"] = 1.19
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)


def test_schema_rejects_numeric_quantity(schema):
    """Quantity is a string so fractional weights survive JSON round-tripping."""
    document, _ = parse_tsv(
        (FIXTURES / "esselunga_b.tsv").read_text(),
        profile_id="esselunga",
        receipt_id="x",
    )
    document["line_items"][0]["quantity"]["value"] = 1.24
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)
