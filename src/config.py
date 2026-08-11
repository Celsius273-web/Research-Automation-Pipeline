"""Runtime configuration for local execution."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Primary bundle directory - papers stored with all artifacts colocated
PAPERS_DIR = DATA_DIR / "papers"
PAPER_BUNDLES_DIR = PAPERS_DIR  # Primary artifact location

# Legacy directories - maintained for backward compatibility
EXTRACTIONS_DIR = DATA_DIR / "extractions"
PLANS_DIR = DATA_DIR / "plans"
RUNS_DIR = DATA_DIR / "runs"
REPORTS_DIR = DATA_DIR / "reports"

# Deterministic experiment results contract (Executor <-> Reviewer).
RESULTS_DIR = DATA_DIR / "results"
RESULTS_SUMMARY_FILENAME = "summary.json"
RESULTS_METRICS_FILENAME = "metrics.csv"
RESULTS_LOGS_FILENAME = "logs.txt"
RESULTS_RUN_DIR_PATTERN = (
    "YYYY-MM-DD_HH-MM-SS__{benchmark_slug}__{method_slug}__seed-{seed}"
)

FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Default model routing for v1 on 16GB unified memory.
ANALYST_MODEL = os.getenv("ANALYST_MODEL", "qwen3.5:9b")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "qwen3.5:9b")
ENGINEER_MODEL = os.getenv("ENGINEER_MODEL", "qwen2.5-coder:7b")
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", "qwen3.5:9b")

# Optional evaluation fallback.
REASONING_FALLBACK_MODEL = os.getenv("REASONING_FALLBACK_MODEL", "gpt-oss:20b")

# Shared generation defaults.
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))
MODEL_NUM_PREDICT = int(os.getenv("MODEL_NUM_PREDICT", "1024"))
PLANNER_NUM_PREDICT = int(os.getenv("PLANNER_NUM_PREDICT", "4096"))

# Maximum characters fed to the analyst per section call.
# Each section can be tens of thousands of chars; 12000 gives ~3 pages of dense
# academic text while staying well within typical context window budgets.
ANALYST_SECTION_CHARS = int(os.getenv("ANALYST_SECTION_CHARS", "12000"))
# Experiments/appendix often hold result tables past the first window; chunk them.
ANALYST_RESULT_SECTION_CHARS = int(os.getenv("ANALYST_RESULT_SECTION_CHARS", "12000"))
ANALYST_RESULT_CHUNK_OVERLAP = int(os.getenv("ANALYST_RESULT_CHUNK_OVERLAP", "800"))
ANALYST_MAX_RESULT_CHUNKS = int(os.getenv("ANALYST_MAX_RESULT_CHUNKS", "4"))
ANALYST_NUM_PREDICT = int(os.getenv("ANALYST_NUM_PREDICT", str(max(MODEL_NUM_PREDICT, 2048))))
ANALYST_MAX_EXTRACTED_TABLES = int(os.getenv("ANALYST_MAX_EXTRACTED_TABLES", "20"))

# Match thresholds used by Reviewer delta classification (match / close / diverged).
REVIEW_MATCH_TOLERANCE_PCT = float(os.getenv("REVIEW_MATCH_TOLERANCE_PCT", "2.0"))
REVIEW_CLOSE_TOLERANCE_PCT = float(os.getenv("REVIEW_CLOSE_TOLERANCE_PCT", "20.0"))

# Agent reasoning and retry configuration.
PLANNER_MAX_RETRIES = int(os.getenv("PLANNER_MAX_RETRIES", "3"))
PLANNER_DEFAULT_SETUP_MINUTES = float(os.getenv("PLANNER_DEFAULT_SETUP_MINUTES", "5"))
PLANNER_MAX_EXAMPLE_COMMANDS = int(os.getenv("PLANNER_MAX_EXAMPLE_COMMANDS", "5"))
PLANNER_MAX_ENTRYPOINT_HINTS = int(os.getenv("PLANNER_MAX_ENTRYPOINT_HINTS", "10"))
PLANNER_MAX_CONTEXT_ITEMS = int(os.getenv("PLANNER_MAX_CONTEXT_ITEMS", "30"))
PLANNER_MAX_NOTE_ITEMS_PER_CATEGORY = int(
    os.getenv("PLANNER_MAX_NOTE_ITEMS_PER_CATEGORY", "8")
)
PLANNER_MAX_NOTE_ITEM_CHARS = int(os.getenv("PLANNER_MAX_NOTE_ITEM_CHARS", "300"))
# Deterministic deep repo exploration for Planner context.
PLANNER_REPO_README_CHARS = int(os.getenv("PLANNER_REPO_README_CHARS", "8000"))
PLANNER_REPO_TREE_DEPTH = int(os.getenv("PLANNER_REPO_TREE_DEPTH", "3"))
PLANNER_REPO_TREE_MAX_ENTRIES = int(os.getenv("PLANNER_REPO_TREE_MAX_ENTRIES", "200"))
PLANNER_REPO_MAX_SOURCE_FILES = int(os.getenv("PLANNER_REPO_MAX_SOURCE_FILES", "12"))
PLANNER_REPO_SOURCE_FILE_CHARS = int(os.getenv("PLANNER_REPO_SOURCE_FILE_CHARS", "2500"))
PLANNER_REPO_EXPLORATION_CHARS = int(os.getenv("PLANNER_REPO_EXPLORATION_CHARS", "24000"))
# Phase-based plan: compact axes + concrete matrix rows (what Engineer runs by default).
PLANNER_PHASE_SEED_COUNT = int(os.getenv("PLANNER_PHASE_SEED_COUNT", "10"))
PLANNER_PHASE_SYNTHETIC_MAX = int(os.getenv("PLANNER_PHASE_SYNTHETIC_MAX", "3"))
PLANNER_PHASE_REALWORLD_MAX = int(os.getenv("PLANNER_PHASE_REALWORLD_MAX", "9"))
PLANNER_PHASE_ALGO_MAX = int(os.getenv("PLANNER_PHASE_ALGO_MAX", "4"))
PLANNER_PHASE_EXAMPLE_ROWS = int(os.getenv("PLANNER_PHASE_EXAMPLE_ROWS", "3"))
PLANNER_PHASE_ABLATION_VALUES_MAX = int(os.getenv("PLANNER_PHASE_ABLATION_VALUES_MAX", "5"))
PLANNER_LIBRARY_TEST_MAX = int(os.getenv("PLANNER_LIBRARY_TEST_MAX", "6"))
PLANNER_LIBRARY_SMOKE_TESTS = int(os.getenv("PLANNER_LIBRARY_SMOKE_TESTS", "3"))
PLANNER_SCRIPT_ENTRYPOINTS_MAX = int(os.getenv("PLANNER_SCRIPT_ENTRYPOINTS_MAX", "12"))
PLANNER_NATIVE_TESTS_MAX = int(os.getenv("PLANNER_NATIVE_TESTS_MAX", "10"))
PLANNER_CONFIG_FILES_MAX = int(os.getenv("PLANNER_CONFIG_FILES_MAX", "20"))
PLANNER_MAKE_TARGETS_MAX = int(os.getenv("PLANNER_MAKE_TARGETS_MAX", "12"))
# Legacy candidate caps kept for exploration helpers.
PLANNER_MATRIX_CANDIDATE_MAX = int(os.getenv("PLANNER_MATRIX_CANDIDATE_MAX", "12"))
PLANNER_MATRIX_MAX_FUNCTIONS = int(os.getenv("PLANNER_MATRIX_MAX_FUNCTIONS", "6"))
PLANNER_MATRIX_MAX_ALGORITHMS = int(os.getenv("PLANNER_MATRIX_MAX_ALGORITHMS", "5"))
PLANNER_MIN_MATRIX_ROWS = int(os.getenv("PLANNER_MIN_MATRIX_ROWS", "6"))
# Planner-generated wrapper/driver stubs live under <paper_bundle>/planner_stubs/.
PLANNER_STUBS_DIRNAME = os.getenv("PLANNER_STUBS_DIRNAME", "planner_stubs")
PLANNER_STUB_EXAMPLE_ROWS = int(os.getenv("PLANNER_STUB_EXAMPLE_ROWS", "4"))

# Default off: Engineer runs plan matrix rows only (fast). Set ENGINEER_EXPAND_FULL_AXES=1
# to expand phase.axes into the full cartesian product (paper-scale suites).
ENGINEER_EXPAND_FULL_AXES = os.getenv("ENGINEER_EXPAND_FULL_AXES", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}

# Engineer/Executor retry loop.
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "5"))
# Plan-driven Engineer CLI retries the same command without LLM patches.
ENGINEER_MAX_ATTEMPTS = int(os.getenv("ENGINEER_MAX_ATTEMPTS", "3"))
EXECUTOR_TIMEOUT_SECONDS = int(os.getenv("EXECUTOR_TIMEOUT_SECONDS", "600"))
EXECUTOR_LOG_MAX_CHARS = int(os.getenv("EXECUTOR_LOG_MAX_CHARS", "20000"))
# Skip a paper when free RAM is below this threshold (16GB hosts get tight under Docker).
MIN_FREE_MEMORY_GB = float(os.getenv("MIN_FREE_MEMORY_GB", "2.0"))
ENGINEER_METRICS_FILENAME = "metrics.json"
ENGINEER_LOG_FILENAME = "engineer.log"
REVIEWER_REPORT_FILENAME = "reviewer_report.json"
# Paper-repo venv created inside the Docker-mounted workspace so installs survive container teardown.
PAPER_VENV_DIRNAME = os.getenv("PAPER_VENV_DIRNAME", ".venv")

# Paper ingestion pipeline.
INGEST_CLONE_TIMEOUT_SECONDS = int(os.getenv("INGEST_CLONE_TIMEOUT_SECONDS", "300"))
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://localhost/research_assistant")
