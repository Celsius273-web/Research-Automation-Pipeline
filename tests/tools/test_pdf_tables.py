from __future__ import annotations

from pathlib import Path

from src.tools.pdf_tables import (
    _reconstruct_row_major_table,
    append_tables_to_sections,
    extract_tables_markdown,
)


def test_reconstruct_row_major_table_builds_markdown() -> None:
    caption = "Table 1: Details of the 9 real-world design optimization problems."
    body = [
        "value.",
        "Problem Name",
        "d",
        "g",
        "g*",
        "f(x*)",
        "Three-bar truss design problem",
        "2",
        "3",
        "1",
        "2.6389E+02",
        "Pressure vessel design",
        "4",
        "4",
        "2",
        "5.8853E+03",
    ]
    markdown = _reconstruct_row_major_table(caption, body)
    assert markdown is not None
    assert "| Problem Name | d | g | g* | f(x*) |" in markdown
    assert "| Three-bar truss design problem | 2 | 3 | 1 | 2.6389E+02 |" in markdown
    assert "| Pressure vessel design | 4 | 4 | 2 | 5.8853E+03 |" in markdown
    assert markdown.splitlines()[0].startswith("### Table 1")


def test_append_tables_to_sections_attaches_to_experiments() -> None:
    updated = append_tables_to_sections(
        {"experiments": "Experiment prose", "appendix": "", "full_text": "Full"},
        "## Extracted Tables\n### Table 1\n| a | b |\n| --- | --- |\n| 1 | 2 |",
    )
    assert "Extracted Tables" in updated["experiments"]
    assert "Experiment prose" in updated["experiments"]
    assert "Extracted Tables" in updated["full_text"]


def test_extract_tables_markdown_finds_boundary_table_one() -> None:
    path = Path("data/papers/boundary_exploration_bo/paper.pdf")
    if not path.exists():
        return
    markdown = extract_tables_markdown(path)
    assert "Extracted Tables" in markdown
    assert "| Problem Name | d | g | g∗ | f(x∗) |" in markdown
    assert "| Three-bar truss design problem | 2 | 3 | 1 | 2.6389E+02 |" in markdown
    assert "| Cantilever beam design | 30 | 21 | 1 | 1.5731E+02 |" in markdown
