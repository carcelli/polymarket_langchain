"""
Text processing utilities based on Fluent Python Ch. 4.
Handles Unicode normalization, diacritic removal, and ASCII-fication.
"""

import unicodedata


def nfc_equal(str1: str, str2: str) -> bool:
    """
    Compare two strings using Normal Form C (NFC).
    Useful because 'café' can be represented by different byte sequences.
    """
    return unicodedata.normalize("NFC", str1) == unicodedata.normalize("NFC", str2)


def fold_equal(str1: str, str2: str) -> bool:
    """
    Compare two strings using Unicode case folding.
    More aggressive than .lower() (e.g., handles German 'ß' -> 'ss').
    """
    return (
        unicodedata.normalize("NFC", str1).casefold()
        == unicodedata.normalize("NFC", str2).casefold()
    )


def shave_marks(txt: str) -> str:
    """
    Remove all diacritic marks (accents, cedillas, etc.).
    Example: 'São Paulo' -> 'Sao Paulo'
    """
    norm_txt = unicodedata.normalize("NFD", txt)
    shaved = "".join(c for c in norm_txt if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", shaved)


# Mapping table for Western typographical symbols to ASCII
# Based on Fluent Python Example 4-17 (dewinize function)
DEWINIZE_MAP = str.maketrans(
    {
        0x80: '"',  # EURO SIGN → quote (or '' if you prefer removal)
        0x81: "",  # unused → remove
        0x82: "'",  # SINGLE LOW-9 QUOTATION MARK
        0x83: '"',  # DOUBLE LOW-9 QUOTATION MARK
        0x84: '"',  # LEFT DOUBLE QUOTATION MARK
        0x85: "...",  # HORIZONTAL ELLIPSIS
        0x86: "-",  # DAGGER
        0x87: "--",  # DOUBLE DAGGER
        0x88: "^",  # CIRCUMFLEX ACCENT (fallback)
        0x89: "%",  # PER MILLE SIGN
        0x8A: "S",  # LATIN CAPITAL LETTER S WITH CARON
        0x8B: "<",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
        0x8C: "OE",  # LATIN CAPITAL LIGATURE OE
        0x8D: "",  # unused
        0x8E: "Z",  # LATIN CAPITAL LETTER Z WITH CARON
        0x8F: "",  # unused
        0x90: "",  # unused
        0x91: "'",  # LEFT SINGLE QUOTATION MARK
        0x92: "'",  # RIGHT SINGLE QUOTATION MARK
        0x93: '"',  # LEFT DOUBLE QUOTATION MARK
        0x94: '"',  # RIGHT DOUBLE QUOTATION MARK
        0x95: "*",  # BULLET
        0x96: "-",  # EN DASH
        0x97: "--",  # EM DASH
        0x98: "~",  # SMALL TILDE
        0x99: "(TM)",  # TRADE MARK SIGN
        0x9A: "s",  # LATIN SMALL LETTER S WITH CARON
        0x9B: ">",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
        0x9C: "oe",  # LATIN SMALL LIGATURE OE
        0x9D: "",  # unused
        0x9E: "z",  # LATIN SMALL LETTER Z WITH CARON
        0x9F: "Y",  # LATIN CAPITAL LETTER Y WITH DIAERESIS
    }
)


def dewinize(txt: str) -> str:
    """Replace Win1252 symbols with ASCII chars or sequences."""
    return txt.translate(DEWINIZE_MAP)


def asciize(txt: str) -> str:
    """
    Transform text to pure ASCII.
    Useful for generating search slugs or cache keys.
    """
    no_marks = shave_marks(dewinize(txt))
    no_marks = no_marks.replace("ß", "ss")
    return unicodedata.normalize("NFKC", no_marks)
