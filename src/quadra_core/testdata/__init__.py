"""Real receipt OCR fixtures, with personal data replaced by placeholders."""

from __future__ import annotations

from pathlib import Path

OCR = Path(__file__).parent / "ocr"


def tsv(name: str) -> str:
    """Read one fixture by name, with or without the .tsv suffix."""
    path = OCR / (name if name.endswith(".tsv") else f"{name}.tsv")
    if not path.is_file():
        raise FileNotFoundError(f"no such fixture: {name}. Have: {names()}")
    return path.read_text()


def names() -> list[str]:
    """Every fixture available, without the suffix."""
    return sorted(p.stem for p in OCR.glob("*.tsv"))
