# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-01

### Added

- `review`: a list of what a person should check, one entry per field, with the item and a plain message.
- `needs_review` on every item and discount, to highlight just those rows.
- Discounts point to the item they apply to, and items show their own `discount_minor`.
- `totals.total_source`: where the total came from (`printed`, `payments` or `supplied`).
- When the total doesn't add up, the review names the lines whose price couldn't be read.
- A price higher than the whole receipt is flagged.
- Crumpled receipts get a second reading with finer lighting correction.
- The price column re-read handles common misreads: a lost leading `1`, a missing comma, `=` instead of `-`.
- Esselunga receipts are recognised even when the header is cut off.
- `OcrError` is exported.
- `--total` takes amounts as printed, e.g. `101,25`.

### Changed

- **Breaking:** the status is now `ok`, `review` or `failed` (`partial` is gone). Exit codes are 0, 1, 2.
- **Breaking:** `parse_image` and `parse_tsv` return a `Result` (`status`, `document`, `tsv`, `image`) instead of a tuple.
- **Breaking:** the arguments are now `profile=` and `total=`, replacing `profile_id=` and `printed_total=`.
- **Breaking:** new JSON schema, `receipt-2.0.0`.
- A price fixed by the re-read is trusted only if the total was confirmed twice and every re-read agrees. Otherwise it goes to review as a suggestion.
- Prices are never adjusted towards a total that the payment line contradicts.
- Warnings no longer change the status.
- Description confidence is the average of its words, not the worst one.
- The same input always gives the same output.
- The price column is re-read three ways, and the readings vote.

### Removed

- The `merchant`, `transaction`, `source` and `receipt_id` fields, which were always empty.
- Review bookkeeping fields (`sampled`, `mode`, `state`, `reviewed_at`…) and the `review=` argument.
- `profile.match_confidence`.
- `no_review`, `read_input`, `load`, `select`, `available` and `Profile` from the top-level package.
- `tools/try_photo.py`.

### Fixed

- A discount that lost its minus sign added money instead of taking it off.
- `di cui IVA` and the VAT legend became items when the total was unreadable.
- A misread total line could become an item.
- Invalid VAT codes like `e` or `i`. An unreadable code is now empty.
- VAT markers and crease marks stuck to descriptions.
- Prices and discounts lost over receipt creases are recovered.
- The `printed_total_not_found` warning stayed even after typing the total in.
- `import quadra_core` failed on Windows.
- The CLI showed a traceback for OCR errors and non-text files.

## [0.1.0] - 2026-09-11

### Added

- First release.

[0.2.0]: https://github.com/SteMazzO/quadra-core/compare/0.1...0.2.0
[0.1.0]: https://github.com/SteMazzO/quadra-core/releases/tag/0.1
