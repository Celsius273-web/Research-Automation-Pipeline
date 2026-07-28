"""Deterministic PDF table extraction for Analyst grounding."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from src.config import ANALYST_MAX_EXTRACTED_TABLES

_TABLE_CAPTION_RE = re.compile(
    r"(?im)^\s*table\s+(\d+[a-z]?)\s*[:.\-–—]\s*(.+?)\s*$"
)
_STOP_LINE_RE = re.compile(
    r"(?im)^(figure\s+\d+|fig\.\s*\d+|algorithm\s+\d+|appendix\s+[a-z0-9]|"
    r"\d+(\.\d+)*\s+[A-Z]|references\b|acknowledg)"
)
_NUMERIC_CELL_RE = re.compile(r"(?i)\d+(\.\d+)?([eE][+-]?\d+)?%?")


def _cell_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _table_fill_ratio(rows: list[list[object]]) -> float:
    cells = [cell for row in rows for cell in row]
    if not cells:
        return 0.0
    filled = sum(1 for cell in cells if _cell_text(cell))
    return filled / len(cells)


def _rows_to_markdown(title: str, rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:] if len(normalized) > 1 else []
    lines = [
        f"### {title}",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _is_usable_detected_table(rows: list[list[object]]) -> bool:
    if len(rows) < 2:
        return False
    if _table_fill_ratio(rows) < 0.45:
        return False
    flat = [_cell_text(cell) for row in rows for cell in row]
    numeric_cells = sum(1 for cell in flat if _NUMERIC_CELL_RE.search(cell))
    # Plot/figure false positives rarely have many clean numeric cells.
    return numeric_cells >= 2


def _extract_detected_tables(doc: fitz.Document) -> list[str]:
    blocks: list[str] = []
    for page_index, page in enumerate(doc, start=1):
        try:
            finder = page.find_tables()
        except Exception:
            continue
        tables = list(getattr(finder, "tables", []) or [])
        for table_index, table in enumerate(tables, start=1):
            raw_rows = table.extract() or []
            if not _is_usable_detected_table(raw_rows):
                continue
            rows = [[_cell_text(cell) for cell in row] for row in raw_rows]
            title = f"Detected table (page {page_index}, #{table_index})"
            markdown = _rows_to_markdown(title, rows)
            if markdown:
                blocks.append(markdown)
            if len(blocks) >= ANALYST_MAX_EXTRACTED_TABLES:
                return blocks
    return blocks


def _looks_like_prose_line(line: str) -> bool:
    """True for narrative sentences that should not be treated as table cells."""
    if len(line) > 80 and not _NUMERIC_CELL_RE.fullmatch(line.replace(" ", "")):
        # Long non-numeric lines are almost always caption wrap or following prose.
        if len(line.split()) >= 8:
            return True
    if len(line) > 100 and not _NUMERIC_CELL_RE.search(line):
        return True
    return False


def _collect_table_body_lines(page_text: str, caption_end: int) -> list[str]:
    """Collect short cell-like lines after a table caption, skipping caption wrap."""
    lines = [line.strip() for line in page_text[caption_end:].splitlines()]
    while lines and (not lines[0] or len(lines[0]) > 100):
        lines.pop(0)

    body: list[str] = []
    for stripped in lines:
        if not stripped:
            if body:
                break
            continue
        if _STOP_LINE_RE.match(stripped):
            break
        if _looks_like_prose_line(stripped):
            break
        body.append(stripped)
        if len(body) >= 80:
            break
    return body


def _score_row_major(headers: list[str], rows: list[list[str]]) -> float:
    """Prefer grids where most rows end in a numeric cell (typical result tables)."""
    if not rows:
        return -1.0
    last_numeric = sum(1 for row in rows if _NUMERIC_CELL_RE.search(row[-1]))
    any_numeric = sum(
        1 for row in rows if any(_NUMERIC_CELL_RE.search(cell) for cell in row)
    )
    if any_numeric < max(1, len(rows) // 2):
        return -1.0
    # Weight last-column numeric density, then prefer wider tables on ties.
    return (last_numeric / len(rows)) * 10.0 + len(headers)


def _try_row_major_from(body_lines: list[str]) -> list[list[str]] | None:
    if len(body_lines) < 4:
        return None
    max_cols = min(8, len(body_lines) // 2)
    best_rows: list[list[str]] | None = None
    best_score = -1.0
    for ncols in range(max_cols, 1, -1):
        header_candidates = body_lines[:ncols]
        if any(_NUMERIC_CELL_RE.search(cell) for cell in header_candidates):
            continue
        remainder = body_lines[ncols:]
        if len(remainder) < ncols or len(remainder) % ncols != 0:
            continue
        rows = [remainder[i : i + ncols] for i in range(0, len(remainder), ncols)]
        score = _score_row_major(header_candidates, rows)
        if score > best_score:
            best_score = score
            best_rows = [header_candidates] + rows
    return best_rows


def _reconstruct_row_major_table(caption: str, body_lines: list[str]) -> str | None:
    """Rebuild tables printed as one cell per line (common in academic PDFs)."""
    # Skip leading caption leftovers ("value.", etc.) before the real header.
    best_rows: list[list[str]] | None = None
    best_score = -1.0
    for start in range(0, min(6, len(body_lines))):
        rows = _try_row_major_from(body_lines[start:])
        if rows is None:
            continue
        score = _score_row_major(rows[0], rows[1:])
        if score > best_score:
            best_score = score
            best_rows = rows
    if best_rows is None:
        return None
    title = caption if caption.lower().startswith("table") else f"Table excerpt: {caption}"
    return _rows_to_markdown(title, best_rows)


def _extract_caption_tables(doc: fitz.Document) -> list[str]:
    blocks: list[str] = []
    seen: set[str] = set()
    for page in doc:
        text = page.get_text("text") or ""
        for match in _TABLE_CAPTION_RE.finditer(text):
            table_id = match.group(1)
            caption_title = match.group(2).strip()
            caption = f"Table {table_id}: {caption_title}"
            body = _collect_table_body_lines(text, match.end())
            if not body:
                continue
            reconstructed = _reconstruct_row_major_table(caption, body)
            if reconstructed:
                key = reconstructed
            else:
                excerpt = "\n".join([caption, *body[:40]])
                key = excerpt
                reconstructed = f"### {caption}\n```\n{excerpt}\n```"
            if key in seen:
                continue
            seen.add(key)
            blocks.append(reconstructed)
            if len(blocks) >= ANALYST_MAX_EXTRACTED_TABLES:
                return blocks
    return blocks


def extract_tables_markdown(pdf_path: str | Path) -> str:
    """Extract usable PDF tables as markdown for Analyst grounding."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    blocks: list[str] = []
    with fitz.open(path) as doc:
        blocks.extend(_extract_detected_tables(doc))
        for block in _extract_caption_tables(doc):
            if block not in blocks:
                blocks.append(block)
            if len(blocks) >= ANALYST_MAX_EXTRACTED_TABLES:
                break

    if not blocks:
        return ""
    return (
        "## Extracted Tables\n"
        "Use these structured tables as the preferred source for reported_results values.\n\n"
        + "\n\n".join(blocks)
    )


def append_tables_to_sections(
    sections_text: dict[str, str],
    tables_markdown: str,
) -> dict[str, str]:
    """Attach extracted tables to result-heavy section text."""
    if not tables_markdown.strip():
        return sections_text
    updated = dict(sections_text)
    for key in ("experiments", "appendix", "full_text"):
        existing = (updated.get(key) or "").strip()
        if existing:
            updated[key] = f"{existing}\n\n{tables_markdown}"
        elif key in {"experiments", "full_text"}:
            updated[key] = tables_markdown
    return updated
