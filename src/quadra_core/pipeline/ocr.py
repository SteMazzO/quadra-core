"""Run Tesseract with resource limits and return its TSV output."""

from __future__ import annotations

import functools
import io
import os
import resource
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Pillow is needed only to OCR an image, never to parse TSV.
    from PIL import Image

# 512MB total. Cap the child well below it so it dies on its own instead of
# letting the OOM killer pick a process.
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
        # Keeps the run of spaces between description and price. Geometry is the
        # main signal, but this makes the raw text usable as a cross-check.
        "-c",
        "preserve_interword_spaces=1",
        # Receipt text is mostly abbreviations and product codes.
        "-c",
        "load_system_dawg=0",
        "-c",
        "load_freq_dawg=0",
        # Without this Tesseract guesses a DPI and can rescale badly. Phone JPEGs
        # often carry no useful resolution metadata.
        "-c",
        "user_defined_dpi=300",
        # Skip the inverted-image retry, since input is always dark on light.
        # This is invert_threshold, not tessedit_do_invert, which 5.3.4
        # deprecates and Tesseract 6 removes.
        "-c",
        "invert_threshold=0",
    ]


def _limit_memory() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


@functools.lru_cache(maxsize=1)
def tesseract_path() -> str:
    """Absolute path to tesseract, or raise if it is not installed.

    The child runs with a fixed PATH, so resolving here and passing the full
    path is what keeps an install outside /usr/bin working: /usr/local/bin and
    Homebrew both used to pass this check and then fail to launch.
    """
    found = shutil.which("tesseract")
    if found is None:
        raise OcrError(
            "tesseract not found on PATH. Install it with: "
            "sudo apt install tesseract-ocr tesseract-ocr-ita"
        )
    return found


@functools.lru_cache(maxsize=1)
def engine_version() -> str:
    """Report the installed Tesseract version, or raise if it is absent.

    Cached: this used to spawn a process on every OCR call, and a photo makes
    three or more of those.
    """
    result = subprocess.run(
        [tesseract_path(), "--version"], capture_output=True, text=True, check=False
    )
    first = (result.stdout or result.stderr).splitlines()
    return first[0].strip() if first else "unknown"


def _child_env(threads: int) -> dict[str, str]:
    """Build a fixed environment, so no ambient locale changes how numbers read."""
    env = {"OMP_THREAD_LIMIT": str(threads), "PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    # Installs outside the Debian layout keep their language data elsewhere and
    # find it through this. Without it they fail with "Error opening data file".
    tessdata = os.environ.get("TESSDATA_PREFIX")
    if tessdata:
        env["TESSDATA_PREFIX"] = tessdata
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
    """OCR an image and return Tesseract's TSV.

    The image goes in on stdin and results come back on stdout, so no temp files.

    threads defaults to 1 because Tesseract's OpenMP support often runs slower on
    small images and uses more memory.
    """
    version = engine_version()

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    try:
        completed = subprocess.run(
            [tesseract_path(), "-", "-", *_base_flags(psm, oem, lang), "tsv"],
            input=buffer.getvalue(),
            capture_output=True,
            timeout=timeout,
            check=False,
            preexec_fn=_limit_memory,
            env=_child_env(threads),
        )
    except subprocess.TimeoutExpired as exc:
        raise OcrError(f"tesseract timed out after {timeout}s") from exc
    except OSError as exc:
        raise OcrError(f"could not run tesseract: {exc}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise OcrError(f"tesseract failed ({completed.returncode}): {detail}")

    return OcrResult(
        tsv=completed.stdout.decode("utf-8", "replace"),
        psm=psm,
        oem=oem,
        lang=lang,
        engine_version=version,
    )


def run_on_path(path: Path, **kwargs) -> tuple[OcrResult, object, object]:
    """Preprocess a photograph and OCR it.

    Returns the prepared image as well. Word boxes are in its coordinates, not the
    original photo's, so anything cropping by box needs this one.
    """
    from .preprocess import preprocess  # noqa: PLC0415 - keeps Pillow optional

    image, report = preprocess(path)
    return run(image, **kwargs), report, image
