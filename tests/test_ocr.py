"""Running Tesseract: the guard rails, not the recognition."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from quadra_core.pipeline import ocr


@pytest.fixture(autouse=True)
def forget_the_cached_path():
    """tesseract_path is cached for the process, and these tests move it about."""
    ocr.tesseract_path.cache_clear()
    yield
    ocr.tesseract_path.cache_clear()


def test_a_missing_tesseract_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(ocr.shutil, "which", lambda _name: None)
    with pytest.raises(ocr.OcrError, match="apt install tesseract-ocr"):
        ocr.tesseract_path()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc to read the limit")
def test_the_child_is_capped_without_a_fork_hook():
    """The cap is set from here; preexec_fn can deadlock a threaded caller."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert ocr._limit_memory(child.pid)
        limits = Path(f"/proc/{child.pid}/limits").read_text()
        capped = next(
            line for line in limits.splitlines() if line.startswith("Max address space")
        )
        assert str(ocr.MEMORY_LIMIT_BYTES) in capped
    finally:
        child.kill()
        child.wait()


def test_capping_a_process_that_already_left_is_not_an_error():
    child = subprocess.Popen([sys.executable, "-c", ""])
    child.wait()
    assert ocr._limit_memory(child.pid) is False


@pytest.mark.skipif(not Path("/bin/false").exists(), reason="needs /bin/false")
def test_a_failing_tesseract_is_reported_not_swallowed(monkeypatch):
    image = pytest.importorskip("PIL.Image")
    monkeypatch.setattr(ocr, "tesseract_path", lambda: "/bin/false")
    monkeypatch.setattr(ocr, "engine_version", lambda: "stub")
    with pytest.raises(ocr.OcrError, match="tesseract failed"):
        ocr.run(image.new("L", (10, 10), 255))


@pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract is not installed"
)
def test_a_slow_receipt_times_out_instead_of_hanging():
    image = pytest.importorskip("PIL.Image")
    with pytest.raises(ocr.OcrError, match="timed out"):
        ocr.run(image.new("L", (400, 400), 255), timeout=0.001)
