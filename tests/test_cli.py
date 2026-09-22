"""Command line tests. The exit code is part of the interface, not just the JSON."""

from __future__ import annotations

import io
import json

from quadra_core.cli import EXIT_ERROR, EXIT_OK, EXIT_PARTIAL, main
from quadra_core.testdata import OCR as FIXTURES


def test_a_clean_receipt_prints_json_and_exits_ok(capsys):
    code = main(["parse", str(FIXTURES / "esselunga_b.tsv"), "--profile", "esselunga"])
    document = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert document["totals"]["balanced"] is True
    assert len(document["line_items"]) == 9


def test_a_receipt_that_does_not_add_up_exits_partial(capsys):
    code = main(
        ["parse", str(FIXTURES / "synthetic_unbalanced.tsv"), "--profile", "synthetic"]
    )
    capsys.readouterr()
    assert code == EXIT_PARTIAL


def test_a_typed_in_total_reaches_the_document(capsys):
    main(
        [
            "parse",
            str(FIXTURES / "esselunga_b.tsv"),
            "--profile",
            "esselunga",
            "--total",
            "9999",
        ]
    )
    document = json.loads(capsys.readouterr().out)
    assert document["totals"]["printed_total_minor"] == 9999


def test_an_unknown_profile_is_an_error_not_a_traceback(capsys):
    code = main(["parse", str(FIXTURES / "esselunga_b.tsv"), "--profile", "nope"])
    assert code == EXIT_ERROR
    assert "no such profile" in capsys.readouterr().err


def test_the_profiles_command_lists_what_is_shipped(capsys):
    code = main(["profiles"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "esselunga" in out
    assert "calibrated" in out


def test_tsv_arrives_on_stdin_too(capsys, monkeypatch):
    text = (FIXTURES / "esselunga_b.tsv").read_text()
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    code = main(["parse", "--profile", "esselunga"])
    document = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert len(document["line_items"]) == 9


def test_the_private_keys_never_reach_the_json(capsys):
    main(["parse", str(FIXTURES / "esselunga_b.tsv"), "--profile", "esselunga"])
    document = json.loads(capsys.readouterr().out)
    assert "_tsv" not in document
    assert "_prepared" not in document
