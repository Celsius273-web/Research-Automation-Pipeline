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

## Paper Bundle Organization

All artifacts for a paper are colocated in a single directory under `data/papers/{paper_id}/`:

```
data/papers/p2p_replication/
├── paper.pdf                         # Original PDF
├── metadata.json                     # Ingestion metadata
├── code/                             # Cloned repository (if provided during ingestion)
├── p2p_replication.json              # Extraction JSON (authoritative)
├── p2p_replication_sections.txt      # Human-readable extraction
├── p2p_replication_plan.json         # Execution plan (created by Planner)
├── report.json                       # Final review report
└── runs/                             # Execution attempts
```

This organization makes it easy for all agents to find related artifacts without searching across multiple directories.

---

## Step 1 — Ingest a paper

Copy a PDF into the project and register it. All files are automatically stored in `data/papers/{paper_id}/`:

```bash
# PDF only
.venv/bin/python scripts/ingest_paper.py --pdf-path "/path/to/paper.pdf"

# PDF + GitHub repository (clones code alongside the PDF)
.venv/bin/python scripts/ingest_paper.py \
  --pdf-path "/path/to/paper.pdf" \
  --repo-url "https://github.com/author/repo" \
  --title "Short descriptive title"
```

**Output:** `data/papers/<paper_id>/paper.pdf`, `metadata.json`, optional `code/`.

---

## Step 2 — Run the Analyst (standalone extraction)

Extracts `research_question`, `methodology`, `datasets`, `variables`, `hyperparameters`,
and `evaluation_metrics` from every section of the paper. Writes extraction to the bundle:

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

**Output files in `data/papers/<paper_id>/`:**

| File | Contents |
|---|---|
| `{paper_id}.json` | Machine-readable JSON with `by_section` + `merged` extraction |
| `{paper_id}_sections.txt` | Human-readable per-section breakdown |

---

## Step 3 — Run the Planner

Requires an **approved** extraction artifact. Produces a step-by-step execution plan the Engineer can follow.

```bash
# From an existing extraction (by paper id)
.venv/bin/python -m src.main plan --paper-id <paper_id>

# Skip the plan review checkpoint
.venv/bin/python -m src.main plan --paper-id <paper_id> --non-interactive
```

**Output:** `data/papers/<paper_id>/{paper_id}_plan.json`

The planner receives:
- Full extraction context (research question, methodology, datasets, hyperparameters, etc.)
- Bundle paths so it understands where artifacts are located
- Repository setup guide (if code was provided)
- Hyperparameter reference extracted from the paper

---

## Step 4 — Full pipeline (analyze + plan)

Runs Analyst then Planner in one go, with human review checkpoints:

```bash
.venv/bin/python -m src.main analyze <paper_id> --with-plan
```

Skip the review prompts with `--non-interactive`.

---

## Step 5 — Execute & review (Phase 3+)

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
# Re-run reviewer on an existing run summary
.venv/bin/python -m src.main review --paper-id <paper_id>
```

---

## Already-ingested papers

| paper_id | Title |
|---|---|
| `pre_trained_gaussian_processes_bayesian_optimization` | Pre-trained Gaussian Processes for Bayesian Optimization |
| `p2p_replication` | P2P Search and Replication |
