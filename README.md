# Quadra core

[![License: MIT](https://img.shields.io/github/license/SteMazzO/quadra-core?logo=Open+Source+Initiative)](https://opensource.org/license/mit)
[![PyPI version](https://img.shields.io/pypi/v/quadra-core?logo=pypi)](https://pypi.org/project/quadra-core/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/quadra-core?logo=pypi)](https://pypi.org/project/quadra-core/)
[![codecov](https://codecov.io/github/SteMazzO/quadra-core/coverage.svg?branch=main)](https://codecov.io/github/SteMazzO/quadra-core?branch=main)

Reads a supermarket till receipt from a photo. No ML model, no network calls: Tesseract does the OCR, then geometry and arithmetic do the rest.

```python
from quadra_core import parse_image

receipt = parse_image("receipt.jpg")

receipt.status      # "ok", "review" or "failed"
receipt.document    # the items, discounts and totals, ready for JSON
```

    pip install quadra-core
    sudo apt install tesseract-ocr tesseract-ocr-ita

## One status to look at

| status | meaning |
|---|---|
| `ok` | Every line adds up to the receipt's total, and nothing on the receipt contradicts it. Nothing to check. |
| `review` | Usable, but a person should check the fields listed in `document["review"]`. Only those. |
| `failed` | No items were found. |

Each review entry names the field, the item it concerns, and what went wrong, in simple terms:

```json
"review": [
  {"code": "total_mismatch", "field": "total", "item": null,
   "message": "The lines add up to 90,09, but the total is 101,25."},
  {"code": "price_missing", "field": "price", "item": null,
   "message": "No price read for “BANANE”."},
  {"code": "price_unconfirmed", "field": "price", "item": 3,
   "message": "Read as 1,98; 1,95 would make the receipt add up."}
]
```

Items with an entry also carry `"needs_review": true`, so a review screen can highlight just those rows. Everything else in `warnings` is for debugging and never changes the status.

## How it decides

The line totals plus the discounts have to equal the total, so the parser can check its own work.

When they don't, it reads the price column again on its own, three different ways, and looks for the prices that make the receipt add up. The arithmetic only *proposes*. A correction is accepted without review only when both of these hold:

- the total was confirmed by a second reading (the printed total and the payment line agree, or you typed it in), and
- every re-read of that price agrees on the new value.

Anything less is kept as a suggestion, and that line goes to review. Prices are never adjusted towards a total that the payment line contradicts.

## When the total is unreadable

Type it in, and every line is checked against it:

```python
receipt = parse_image("receipt.jpg", total=10125)   # minor units: 101,25
```

## Command line

    quadra-core parse receipt.jpg                 # prints the JSON
    quadra-core parse receipt.jpg --total 101,25
    quadra-core profiles                          # the shops supported

The exit code is the status: 0 ok, 1 review, 2 failed, 3 error.

## The document

| key | what it holds |
|---|---|
| `status`, `review` | as above |
| `line_items` | description, quantity (a decimal string), unit price, line total, the discount printed under it, VAT code, and where it sits on the photo |
| `adjustments` | the discounts, each pointing at the item it applies to |
| `totals` | subtotal, discounts, the computed total, the total it was checked against and where that came from (`printed`, `payments` or `supplied`), and the difference |
| `warnings`, `ocr` | debugging detail |

Money is always in integer minor units (cents). A field the parser could not read is `null`; it is never guessed. [The JSON Schema](src/quadra_core/schema/receipt-2.0.0.schema.json) is exported as `quadra_core.SCHEMA_PATH`.

`receipt.tsv` and `receipt.image` hold the OCR and the prepared photo that item boxes refer to, if you want to archive them. You can re-parse the TSV later with `parse_tsv`, without the photo.

## Adding a shop

Everything shop-specific is a TOML profile in `src/quadra_core/profiles/`. The code knows how to find a price column, and the profile says what a price looks like.

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

It prints what Tesseract read, where the price column landed, which rule matched each line, and what a person would be asked to review. Add `[[rules]]` and `[regions]` anchors until the lines stop coming back as `unknown`. Rules are tried in order and the first match wins, so put the specific ones first.

Add `[fingerprint]` anchors so the shop is recognised without `--profile`. A VAT number is more stable than a branch address. Save the OCR as a fixture with `--save-fixture NAME` so later changes have something to regress against. Set `calibrated = true` once it balances, otherwise every parse carries a `profile_not_calibrated` warning.

Before committing a fixture, check it for personal data: the address, the loyalty card number, the points balance. `tools/scrub.py` covers the Esselunga ones.

## Development

    uv run pytest

## Licence

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
