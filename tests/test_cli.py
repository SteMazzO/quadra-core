"""Command line tests. The exit code is part of the interface, not just the JSON."""

from __future__ import annotations

import argparse
import io
import json

import pytest

from quadra_core.cli import EXIT_ERROR, EXIT_OK, EXIT_REVIEW, amount, main
from quadra_core.testdata import OCR as FIXTURES


def test_a_clean_receipt_prints_json_and_exits_ok(capsys):
    code = main(["parse", str(FIXTURES / "esselunga_b.tsv")])
    document = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert document["totals"]["balanced"] is True
    assert len(document["line_items"]) == 9


def test_a_receipt_that_does_not_add_up_exits_review(capsys):
    code = main(["parse", str(FIXTURES / "synthetic_unbalanced.tsv")])
    capsys.readouterr()
    assert code == EXIT_REVIEW


def test_a_typed_in_total_reaches_the_document(capsys):
    main(
        [
            "parse",
            str(FIXTURES / "esselunga_b.tsv"),
            "--total",
            "99,99",
        ]
    )
    document = json.loads(capsys.readouterr().out)
    assert document["totals"]["total_minor"] == 9999
    assert document["totals"]["total_source"] == "supplied"


def test_tsv_arrives_on_stdin_too(capsys, monkeypatch):
    text = (FIXTURES / "esselunga_b.tsv").read_text()
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    code = main(["parse"])
    document = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert len(document["line_items"]) == 9


@pytest.mark.parametrize(
    "text,cents", [("101,25", 10125), ("101.25", 10125), ("101", 10100), ("0,1", 10)]
)
def test_a_total_is_typed_the_way_it_is_printed(text, cents):
    assert amount(text) == cents


@pytest.mark.parametrize("text", ["0", "-5", "1,234", "abc"])
def test_a_total_that_is_not_an_amount_is_refused(text):
    with pytest.raises(argparse.ArgumentTypeError):
        amount(text)


def test_a_file_that_is_not_text_is_an_error_not_a_traceback(tmp_path, capsys):
    photo = tmp_path / "receipt.heic"
    photo.write_bytes(b"\xff\xd8\xff\xe0 not text")
    assert main(["parse", str(photo)]) == EXIT_ERROR
    assert capsys.readouterr().err.startswith("error:")
