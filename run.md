# ResearchAssistant Usage

## Prerequisites

**Database setup (SQLite, run once):**
```bash
.venv/bin/python scripts/ingest_paper.py --init-db
```

**Ollama must be running with the analyst model loaded:**
```bash
ollama serve          # in a separate terminal if not already running
ollama pull qwen3.5:9b
```

---

## Step 1 — Ingest a paper

Copy a PDF into the project and register it:

```bash
# PDF only
.venv/bin/python scripts/ingest_paper.py --pdf-path "/path/to/paper.pdf"

# PDF + GitHub repository (clones code alongside the PDF)
.venv/bin/python scripts/ingest_paper.py \
  --pdf-path "/path/to/paper.pdf" \
  --repo-url "https://github.com/author/repo" \
  --title "Short descriptive title"
```

Output: `data/papers/<paper_id>/paper.pdf`, `metadata.json`, optional `code/`.

---

## Step 2 — Run the Analyst (standalone extraction)

Extracts `research_question`, `methodology`, `datasets`, `variables`, `hyperparameters`,
and `evaluation_metrics` from every section of the paper. Writes two files to
`data/extractions/`:

```bash
.venv/bin/python -c "
from src.tools.pdf_parser import parse_pdf_sections
from src.agents.analyst import PaperAnalyst
from src.persistence import persist_extraction_bundle
from src.state import PaperMetadata, ReviewRecord

paper_id = 'YOUR_PAPER_ID'   # folder name under data/papers/

sections = parse_pdf_sections(f'data/papers/{paper_id}/paper.pdf')
analyst  = PaperAnalyst()
bundle   = analyst.extract(sections)

paper  = PaperMetadata(
    paper_id = paper_id,
    title    = 'Your Paper Title',
    pdf_path = f'data/papers/{paper_id}/paper.pdf',
)
review   = ReviewRecord(status='approved')
json_path = persist_extraction_bundle(paper, bundle, review)
txt_path  = json_path.parent / (json_path.stem + '_sections.txt')

print(txt_path.read_text(encoding='utf-8'))
print('Saved:', json_path)
"
```

**Output files:**

| File | Contents |
|---|---|
| `data/extractions/<paper_id>.json` | Machine-readable JSON with `by_section` + `merged` |
| `data/extractions/<paper_id>_sections.txt` | Human-readable per-section breakdown |

---

## Step 3 — Full pipeline (analyze + plan)

Runs the analyst interactively through the CLI with human review checkpoints:

```bash
.venv/bin/python -m src.main analyze <paper_id> --with-plan
```

Skip the review prompts with `--non-interactive`.

---

## Step 4 — Execute & review (Phase 3+)

```bash
# Execute plan + run reviewer
.venv/bin/python -m src.main execute --paper-id <paper_id> --with-review

# Execute against a different local repo
.venv/bin/python -m src.main execute --paper-id <paper_id> \
  --repo-path "/path/to/different/repo" --with-review
```

---

## Standalone commands

```bash
# Re-run planner on an existing extraction
.venv/bin/python -m src.main plan --paper-id <paper_id>

# Re-run reviewer on an existing run summary
.venv/bin/python -m src.main review --paper-id <paper_id>
```

---

## Already-ingested papers

| paper_id | Title |
|---|---|
| `pre_trained_gaussian_processes_bayesian_optimization` | Pre-trained Gaussian Processes for Bayesian Optimization |
| `p2p_replication` | P2P Search and Replication |