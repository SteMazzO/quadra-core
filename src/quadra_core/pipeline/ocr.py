"""Run Tesseract with resource limits and return its TSV output."""

from __future__ import annotations

import functools
import io
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Pillow is only needed to OCR an image.
    from PIL import Image

# Well under a 512MB box, so Tesseract fails by itself before the OOM killer acts.
MEMORY_LIMIT_BYTES = 320 * 1024 * 1024
TIMEOUT_SECONDS = 60


class OcrError(RuntimeError):
    """Tesseract was missing, failed, or timed out."""


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Tesseract output plus the settings that produced it, recorded per receipt."""

    tsv: str
    psm: int
    oem: int
    lang: str
    engine_version: str


def _base_flags(psm: int, oem: int, lang: str) -> list[str]:
    return [
        "--oem",
        str(oem),
        "--psm",
        str(psm),
        "-l",
        lang,
        "-c",
        "preserve_interword_spaces=1",
        # Receipts are mostly abbreviations and codes, so skip the dictionaries.
        "-c",
        "load_system_dawg=0",
        "-c",
        "load_freq_dawg=0",
        # Phone JPEGs rarely carry a usable DPI.
        "-c",
        "user_defined_dpi=300",
        # No inverted-image retry: input is always dark on light.
        "-c",
        "invert_threshold=0",
    ]


def _limit_memory(pid: int) -> bool:
    """Cap a running child's address space. Returns whether the cap was set."""
    try:
        import resource  # noqa: PLC0415 - Unix only
    except ImportError:  # pragma: no cover - Windows
        return False
    prlimit = getattr(resource, "prlimit", None)
    if prlimit is None:  # pragma: no cover - not Linux
        return False
    try:
        prlimit(pid, resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ProcessLookupError, PermissionError):
        # It already exited, or it is not ours to limit.
        return False
    return True


@functools.lru_cache(maxsize=1)
def tesseract_path() -> str:
    """Absolute path to tesseract, or raise if it is not installed."""
    # Resolved here because the child process runs with a fixed PATH.
    found = shutil.which("tesseract")
    if found is None:
        raise OcrError(
            "tesseract not found on PATH. Install it with: "
            "sudo apt install tesseract-ocr tesseract-ocr-ita"
        )
    return found


@functools.lru_cache(maxsize=1)
def engine_version() -> str:
    """Report the installed Tesseract version, or raise if it is absent."""
    result = subprocess.run(
        [tesseract_path(), "--version"], capture_output=True, text=True, check=False
    )
    first = (result.stdout or result.stderr).splitlines()
    return first[0].strip() if first else "unknown"


def _child_env(threads: int) -> dict[str, str]:
    """Build a fixed environment, so no ambient locale changes how numbers read."""
    env = {"OMP_THREAD_LIMIT": str(threads), "PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    # TESSDATA_PREFIX for installs that keep language data elsewhere; SYSTEMROOT
    # because Windows programs cannot start without it.
    for name in ("TESSDATA_PREFIX", "SYSTEMROOT"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    return env


def run(
    image: Image.Image,
    *,
    psm: int = 4,
    oem: int = 1,
    lang: str = "ita",
    timeout: int = TIMEOUT_SECONDS,
    threads: int = 1,
) -> OcrResult:
    """OCR an image and return Tesseract's TSV."""
    version = engine_version()

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    command = [tesseract_path(), "-", "-", *_base_flags(psm, oem, lang), "tsv"]
    try:
        with subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_child_env(threads),
        ) as child:
            _limit_memory(child.pid)
            try:
                stdout, stderr = child.communicate(buffer.getvalue(), timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                child.kill()
                child.communicate()
                raise OcrError(f"tesseract timed out after {timeout}s") from exc
    except OSError as exc:
        raise OcrError(f"could not run tesseract: {exc}") from exc

    if child.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()
        raise OcrError(f"tesseract failed ({child.returncode}): {detail}")

    return OcrResult(
        tsv=stdout.decode("utf-8", "replace"),
        psm=psm,
        oem=oem,
        lang=lang,
        engine_version=version,
    )


def run_on_path(path: Path, **kwargs) -> tuple[OcrResult, object, object]:
    """Preprocess and OCR a photo; returns (result, report, prepared image)."""
    from .preprocess import preprocess  # noqa: PLC0415 - keeps Pillow optional

    image, report = preprocess(path)
    return run(image, **kwargs), report, image
