"""Real receipt OCR, kept with the library rather than with the tests.

This is the corpus the Esselunga profile was calibrated against. It ships so
anyone writing a profile for another shop has known-good output to compare
against.

The address, loyalty number and points balance are replaced with placeholders.
The shop name, VAT number, products and prices are real, and so is every
bounding box, since the geometry is what the tests check.
"""

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
