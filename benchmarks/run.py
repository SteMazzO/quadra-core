"""Score quadra-core on a public receipt dataset.

    uv run --group bench python benchmarks/run.py cord --split validation
    uv run --group bench python benchmarks/run.py cord --failures out/

`--engine truth` feeds the dataset's own transcription to the parser, which
scores the parsing alone. Any other engine reads the photos.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import DATASETS
from scoring import Sample, score, summarise

from quadra_core import parse_image, parse_tsv

TSV_HEADER = "\t".join(
    [
        *("level", "page_num", "block_num", "par_num", "line_num", "word_num"),
        *("left", "top", "width", "height", "conf", "text"),
    ]
)


def to_tsv(words) -> str:
    """Write ground-truth words the way Tesseract would."""
    rows = []
    for text, left, top, width, height in words:
        # A transcribed box can hold several words: share its width by length.
        step = width / max(1, len(text))
        start = 0
        for part in text.split():
            start = text.index(part, start)
            rows.append(
                f"5\t1\t1\t1\t1\t{len(rows) + 1}\t{round(left + start * step)}\t{top}"
                f"\t{round(len(part) * step)}\t{height}\t100\t{part}"
            )
            start += len(part)
    return "\n".join([TSV_HEADER, *rows]) + "\n"


def read(sample: Sample, engine: str, cache: Path) -> dict:
    """Parse one sample, from its ground-truth words or from its photo."""
    if engine == "truth":
        return parse_tsv(to_tsv(sample.words)).document
    photo = cache / "photos" / f"{sample.id}.png"
    if not photo.exists():
        from PIL import Image  # noqa: PLC0415

        photo.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(sample.image)).save(photo)
    return parse_image(photo).document


def main() -> None:
    """Run the benchmark and print its summary as JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--engine", default="truth")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache/quadra-bench")
    ap.add_argument("--failures", type=Path, help="write each wrong receipt here")
    args = ap.parse_args()

    outcomes = []
    for count, sample in enumerate(DATASETS[args.dataset](args.cache, args.split)):
        if args.limit is not None and count >= args.limit:
            break
        document = read(sample, args.engine, args.cache)
        outcome = score(sample, document)
        outcomes.append(outcome)
        if args.failures and not outcome.exact:
            args.failures.mkdir(parents=True, exist_ok=True)
            (args.failures / f"{sample.id}.json").write_text(
                json.dumps(
                    {"truth": sample.truth, "document": document},
                    indent=1,
                    default=lambda o: (
                        getattr(o, "__dict__", None)
                        or {k: getattr(o, k) for k in o.__slots__}
                        if hasattr(o, "__slots__")
                        else str(o)
                    ),
                    ensure_ascii=False,
                )
            )
    print(json.dumps(summarise(outcomes), indent=1))


if __name__ == "__main__":
    main()
