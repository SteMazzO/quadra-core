# Quadra core

[![License: MIT](https://img.shields.io/github/license/SteMazzO/quadra-core?logo=Open+Source+Initiative)](https://opensource.org/license/mit)
[![PyPI version](https://img.shields.io/pypi/v/quadra-core?logo=pypi)](https://pypi.org/project/termtypr/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/quadra-core?logo=pypi)](https://pypi.org/project/termtypr/)
[![codecov](https://codecov.io/github/SteMazzO/quadra-core/coverage.svg?branch=main)](https://codecov.io/github/SteMazzO/termtypr?branch=main)

Reads a supermarket till receipt from a photo. No ML model, no network calls: Tesseract does the OCR, then geometry and arithmetic do the rest.

```python
from pathlib import Path
from quadra_core import parse_image

document, status = parse_image(Path("receipt.jpg"), profile_id="esselunga")
```

`status` is `ok`, `partial` or `failed`. It never raises just because a receipt was hard to read: you get a document back either way, with whatever went wrong listed in `validation.warnings`.

    pip install quadra-core
    sudo apt install tesseract-ocr tesseract-ocr-ita   # for images

## How it works

The line totals plus any discounts have to equal the printed total, so the parser can check its own work.

When the sums don't match it crops out the price column and OCRs that on its own, which gives a second independent reading of every price. Where the two readings disagree it keeps both and picks the combination that hits the printed total. That also recovers lines whose price was unreadable the first time. If two different combinations both add up it changes nothing and flags the receipt.

## Adding a shop

Everything shop-specific is a TOML profile in `src/quadra_core/profiles/`. The code knows how to find a price column, the profile says what a price looks like.

Start with a stub, since `calibrate.py` needs a profile to report against:

```toml
# src/quadra_core/profiles/myshop.toml
[profile]
id = "myshop"
version = 1
calibrated = false

[format]
money = '\d{1,3},\d{2}'
```

Then point it at a photo:

    python tools/calibrate.py receipt.jpg --profile myshop

It prints what Tesseract read, where the price column landed, which rule matched each line, and whether the totals add up. Add `[[rules]]` and `[regions]` anchors until the lines stop coming back as `unknown`. Rules are tried in order and the first match wins, so put the specific ones first.

Add `[fingerprint]` anchors so the shop is recognised without `--profile`. A VAT number is more stable than a branch address. Save the OCR as a fixture with `--save-fixture NAME` so later changes have something to regress against. Set `calibrated = true` once it balances, otherwise every parse carries a `profile_not_calibrated` warning.

Before committing a fixture, check it for personal data: the address, the loyalty card number, the points balance. `tools/scrub.py` covers the Esselunga ones.

## Development

    uv run pytest

## Licence

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
