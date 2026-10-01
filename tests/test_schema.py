"""Parser output must validate against the published JSON Schema."""

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
    document = parse_tsv(
        (FIXTURES / f"{name}.tsv").read_text(), profile=profile
    ).document
    jsonschema.Draft202012Validator(schema).validate(document)


def test_schema_rejects_float_money(schema):
    """Money must be integer minor units."""
    document = esselunga_b()
    document["line_items"][0]["line_total_minor"] = 1.19
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)


def test_schema_rejects_numeric_quantity(schema):
    """Quantity must be a string, so fractions survive JSON."""
    document = esselunga_b()
    document["line_items"][0]["quantity"] = 1.24
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)


def test_schema_rejects_fields_it_does_not_define(schema):
    """A typo in the builder must fail here, not reach users as a stray key."""
    document = esselunga_b()
    document["reviewed"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)


def esselunga_b():
    return parse_tsv(
        (FIXTURES / "esselunga_b.tsv").read_text(), profile="esselunga"
    ).document
