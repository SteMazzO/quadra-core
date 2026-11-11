"""The words receipts use for their totals, in the languages written in Latin script.

Only the lines below the items are labelled this way. What a labelled amount
means for the sum is decided by the arithmetic, not by the word.
"""

from __future__ import annotations

import unicodedata
from itertools import pairwise

# Role -> phrases. A phrase matches when each of its words is on the line.
# When several roles match, the longest phrase wins, then the earliest role here.
PHRASES: dict[str, tuple[str, ...]] = {
    "change": (
        "change",
        "kembali",
        "kembalian",
        "resto",
        "vuelto",
        "cambio",
        "rendu",
        "a rendre",
        "ruckgeld",
        "zuruck",
        "wechselgeld",
        "wisselgeld",
        "troco",
        "reszta",
        "para ustu",
        "total kembalian",
        "cash change",
        "change due",
    ),
    "payment": (
        "cash",
        "tunai",
        "card",
        "visa",
        "mastercard",
        "maestro",
        "amex",
        "debit",
        "credit",
        "tendered",
        "paid",
        "payment",
        "pagamento",
        "contanti",
        "bancomat",
        "carta",
        "efectivo",
        "tarjeta",
        "especes",
        "carte",
        "gegeben",
        "bargeld",
        "kartenzahlung",
        "girocard",
        "dinheiro",
        "cartao",
        "contant",
        "gotowka",
        "karta",
        "nakit",
        "bayar",
        "uang bayar",
        "pagato",
        "pagado",
        "paye",
    ),
    "count": (
        "items",
        "item count",
        "total items",
        "total qty",
        "total item",
        "numero articoli",
        "articoli",
        "pezzi",
        "anzahl",
        "articles",
        "articulos",
        "jumlah item",
        "qty",
    ),
    "tax": (
        "tax",
        "vat",
        "gst",
        "iva",
        "tva",
        "mwst",
        "ust",
        "btw",
        "pajak",
        "ppn",
        "pb1",
        "sst",
        "igv",
        "kdv",
        "ptu",
        "hst",
        "pst",
        "impuesto",
        "steuer",
        "imposta",
        "di cui",
        "total tax",
        "tax included",
        "p.rest",
        "pjk",
    ),
    "service": (
        "service",
        "svc",
        "charge",
        "chg",
        "serv",
        "servizio",
        "coperto",
        "servicio",
        "bedienung",
        "tip",
        "tips",
        "gratuity",
        "propina",
        "mancia",
        "trinkgeld",
        "pourboire",
    ),
    "discount": (
        "discount",
        "disc",
        "diskon",
        "potongan",
        "sconto",
        "descuento",
        "remise",
        "rabatt",
        "korting",
        "desconto",
        "rabat",
        "upust",
        "indirim",
        "savings",
        "coupon",
        "promo",
        "voucher",
        "hemat",
        "total discount",
        "total savings",
    ),
    "rounding": (
        "rounding",
        "pembulatan",
        "arrotondamento",
        "redondeo",
        "arrondi",
        "rundung",
        "afronding",
        "arredondamento",
    ),
    "subtotal": (
        "subtotal",
        "sub total",
        "subttl",
        "subtotale",
        "sous total",
        "soustotal",
        "zwischensumme",
        "subtotaal",
        "ara toplam",
        "sub-total",
    ),
    "total": (
        "total",
        "totale",
        "grand total",
        "total due",
        "amount due",
        "balance due",
        "summe",
        "gesamt",
        "zu zahlen",
        "totaal",
        "te betalen",
        "suma",
        "razem",
        "do zaplaty",
        "toplam",
        "jumlah",
        "total bayar",
        "montant",
        "a payer",
        "importe",
        "total euro",
        "total eur",
        "amount",
    ),
}

# The words printed over the columns of items: 'IVA EURO', 'Qty Description Amount'.
HEADINGS = frozenset(
    [
        "euro",
        "eur",
        "iva",
        "vat",
        "price",
        "prezzo",
        "amount",
        "qty",
        "quantity",
        "item",
        "items",
        "description",
        "descrizione",
        "importo",
        "total",
        "betrag",
        "menge",
        "artikel",
        "preis",
        "prix",
        "montant",
        "designation",
        "precio",
        "cantidad",
        "importe",
        "harga",
        "jumlah",
        "nama",
        "unit",
        "um",
        "rp",
        "rm",
    ]
)

# Only words this long may differ from a keyword, and by one character.
_FUZZY_MIN_LENGTH = 5

# Digits OCR puts where a word has letters: 'T0TALE'.
_LETTERS = str.maketrans("0158|", "oissl")


def _plain(word: str) -> str:
    """Lower-case, unaccented, with punctuation dropped from both ends."""
    word = unicodedata.normalize("NFKD", word.casefold())
    word = "".join(c for c in word if not unicodedata.combining(c))
    word = word.strip(" :;,*=()[]<>!?\"'")
    if sum(c.isalpha() for c in word) >= 2:
        word = word.translate(_LETTERS)
    return word


def _same(word: str, keyword: str) -> bool:
    """Whether a printed word is a keyword, give or take one misread character."""
    if word == keyword:
        return True
    short, long = sorted((word, keyword), key=len)
    if len(short) < _FUZZY_MIN_LENGTH or len(long) - len(short) > 1:
        return False
    # One character replaced, or one dropped.
    return any(
        short[:i] + short[i + (len(long) == len(short)) :] == long[:i] + long[i + 1 :]
        for i in range(len(long))
    )


_COMPILED = [
    (role, tuple(_plain(word) for word in phrase.split()))
    for role, phrases in PHRASES.items()
    for phrase in phrases
]


def is_heading(text: str) -> bool:
    """Whether a line is nothing but column headings."""
    words = [plain for word in text.split() if (plain := _plain(word))]
    return bool(words) and all(word in HEADINGS for word in words)


def role(text: str) -> str | None:
    """Say what a footer line is by its label, or None if no keyword is on it."""
    words = [plain for word in text.split() if (plain := _plain(word))]
    # 'Sub Total' and 'SubTotal' are the same label.
    words += [a + b for a, b in pairwise(words)]

    def position(part: str) -> int:
        return next((i for i, word in enumerate(words) if _same(word, part)), -1)

    # The longest phrase wins, then the one read exactly, then the one printed first:
    # 'Pajak Resto' is a tax in Indonesian before it is change in Italian.
    best, found = None, None
    for name, phrase in _COMPILED:
        places = [position(part) for part in phrase]
        if -1 not in places:
            rank = (len(phrase), sum(part in words for part in phrase), -min(places))
            if best is None or rank > best:
                best, found = rank, name
    return found
