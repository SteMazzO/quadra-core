"""Print what the parser saw and decided on chosen samples, beside the truth.

uv run --group bench python benchmarks/show.py cord validation 14 49
uv run --group bench python benchmarks/show.py cord validation --wrong
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import DATASETS
from run import to_tsv
from scoring import score

from quadra_core import parse_tsv
from quadra_core.pipeline.lexicon import role
from quadra_core.pipeline.lines import drop_speckle, group_lines, load_tsv


def show(sample) -> None:
    """Print one sample: its lines, the parsed document and the truth."""
    tsv = to_tsv(sample.words)
    document = parse_tsv(tsv).document
    outcome = score(sample, document)
    totals = document["totals"]
    print(
        f"===== {sample.id} {document['status']} exact={outcome.exact} {outcome.fields}"
    )
    for line in group_lines(drop_speckle(load_tsv(tsv))):
        print(f"  {line.index:3d} {role(line.text) or '':9s}| {line.text}")
    for item in document["line_items"]:
        sum_ = f"{item['quantity']:>5} x {item['unit_price_minor']}"
        print(f"  ITEM {item['line_index']:3d} {sum_} = {item['line_total_minor']}")
        print(f"       {item['description']}")
    print(
        "  ADJ", [(a["line_index"], a["amount_minor"]) for a in document["adjustments"]]
    )
    print(
        "  CHG",
        [(c["kind"], c["amount_minor"], c["applied"]) for c in document["charges"]],
    )
    print(
        "  TOT", totals, [r["code"] for r in document["review"]], document["warnings"]
    )
    truth = sample.truth
    print(
        "  TRUE",
        [(i.name, str(i.quantity), i.total) for i in truth.items],
        "total",
        truth.total,
        "sub",
        truth.subtotal,
        "tax",
        truth.tax,
    )


def brief(sample) -> None:
    """Print one line per wrong receipt: what differs."""
    document = parse_tsv(to_tsv(sample.words)).document
    outcome = score(sample, document)
    got = sorted(i["line_total_minor"] for i in document["line_items"])
    want = sorted(i.total for i in sample.truth.items)
    print(
        f"{sample.id} {document['status']} {outcome.fields} got={got} want={want}"[:260]
    )


def main() -> None:
    """Show the samples named on the command line, or every wrong one."""
    dataset, split, *wanted = sys.argv[1:]
    cache = Path.home() / ".cache/quadra-bench"
    for index, sample in enumerate(DATASETS[dataset](cache, split)):
        if str(index) in wanted or (
            "--wrong" in wanted
            and not score(sample, parse_tsv(to_tsv(sample.words)).document).exact
        ):
            (brief if "--brief" in wanted else show)(sample)


if __name__ == "__main__":
    main()
