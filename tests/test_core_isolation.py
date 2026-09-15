"""Parsing TSV must not import Pillow, which is slow to load on small devices."""

from __future__ import annotations

import builtins

import pytest

from quadra_core.testdata import OCR as FIXTURES


@pytest.fixture
def no_pillow(monkeypatch):
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] == "PIL":
            raise AssertionError(f"parsing core imported PIL via {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)


def test_tsv_parses_without_pillow(no_pillow):
    # Imported inside the guard so the import itself is what gets checked.
    from quadra_core import parse_tsv  # noqa: PLC0415

    doc, status = parse_tsv(
        (FIXTURES / "esselunga_b.tsv").read_text(),
        profile_id="esselunga",
        receipt_id="iso",
    )
    assert status == "ok"
    assert len(doc["line_items"]) == 9
    assert doc["adjustments"][0]["amount_minor"] == -720
