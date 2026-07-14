"""PDF parsing and section splitting for Paper Analyst."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from src.state import SECTION_NAMES, SectionTextMap


HEADING_PATTERNS: dict[str, list[str]] = {
    "abstract": [
        r"^\s*abstract\s*$",
    ],
    "method": [
        # Standalone keyword (short/simple papers)
        r"^\s*(method|methodology|approach|proposed method)\s*$",
        # Numbered top-level section whose title begins with a method keyword
        r"^\s*\d+\.?\s+(method|methodology|approach|pre.training|framework|algorithm)\b[^\n]{0,60}$",
    ],
    "experiments": [
        r"^\s*(experiments?|experimental setup)\s*$",
        r"^\s*\d+\.?\s+(experiments?|experimental setup)\b[^\n]{0,60}$",
    ],
    "hyperparameters": [
        # Appendix with experiment/training details is the richest source; try first
        r"^\s*appendix\s+\w+[\.\s]+(hyperparameters?|training details|implementation details|experiment details)\b[^\n]{0,60}$",
        # Numbered section with explicit keyword
        r"^\s*\d+\.?\s+(hyperparameters?|training details|implementation details|experiment details)\b[^\n]{0,60}$",
        # Standalone keyword only as last resort (may match table column headers)
        r"^\s*(hyperparameters?|training details|implementation details)\s*$",
    ],
    "appendix": [
        r"^\s*(appendix|supplementary material|supplement)\b[^\n]{0,60}$",
    ],
}

# Fallback starting positions (as fractions of document length) for sections
# whose headings aren't found by pattern matching. Chosen so each section call
# sees a different part of the paper rather than always the document start.
FALLBACK_SECTION_STARTS: dict[str, float] = {
    "abstract": 0.0,
    "method": 0.12,
    "experiments": 0.45,
    "hyperparameters": 0.78,
    "appendix": 0.88,
}


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return normalize_text("\n\n".join(pages))


def _is_toc_line(text: str, match_end: int) -> bool:
    """Return True if the match is a table-of-contents entry rather than a body heading.

    TOC entries are followed either by a bare page number or by a dotted leader
    (". . . . 28"). Body headings are followed by paragraph text.
    """
    after = text[match_end: match_end + 120]
    first_nonempty = next((ln.strip() for ln in after.split("\n") if ln.strip()), "")
    # Bare page number
    if re.match(r"^\d{1,4}$", first_nonempty):
        return True
    # Dotted leader line (e.g. ". . . . . . . 28")
    if re.match(r"^[.\s]{3,}", first_nonempty):
        return True
    return False


def _find_section_positions(text: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    for section in SECTION_NAMES:
        for pattern in HEADING_PATTERNS[section]:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
                if not _is_toc_line(text, match.end()):
                    positions[section] = match.start()
                    break
            if section in positions:
                break
    return positions


def _apply_window_fallbacks(positions: dict[str, int], total_chars: int) -> dict[str, int]:
    """Fill in position estimates for any section not found by heading detection.

    Uses document-fraction anchors so each section call receives a distinct
    slice of text rather than defaulting to the document start every time.
    """
    filled = dict(positions)
    for section in SECTION_NAMES:
        if section not in filled:
            filled[section] = int(FALLBACK_SECTION_STARTS[section] * total_chars)
    return filled


def split_sections(text: str) -> SectionTextMap:
    full_text = normalize_text(text)
    if not full_text:
        return SectionTextMap()

    detected = _find_section_positions(full_text)
    positions = _apply_window_fallbacks(detected, len(full_text))

    ordered = sorted(positions.items(), key=lambda item: item[1])
    slices: dict[str, str] = {}

    for idx, (name, start) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(full_text)
        slices[name] = normalize_text(full_text[start:end])

    data = {name: slices.get(name, "") for name in SECTION_NAMES}
    data["full_text"] = full_text
    return SectionTextMap(**data)


def parse_pdf_sections(pdf_path: str | Path) -> SectionTextMap:
    raw_text = extract_pdf_text(pdf_path)
    return split_sections(raw_text)
