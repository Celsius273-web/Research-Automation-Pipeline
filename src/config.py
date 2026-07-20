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

# Maximum characters fed to the analyst per section call.
# Each section can be tens of thousands of chars; 12000 gives ~3 pages of dense
# academic text while staying well within typical context window budgets.
ANALYST_SECTION_CHARS = int(os.getenv("ANALYST_SECTION_CHARS", "12000"))

REVIEW_MATCH_TOLERANCE_PCT = float(os.getenv("REVIEW_MATCH_TOLERANCE_PCT", "5.0"))
REVIEW_CLOSE_TOLERANCE_PCT = float(os.getenv("REVIEW_CLOSE_TOLERANCE_PCT", "20.0"))

# Agent reasoning and retry configuration.
PLANNER_MAX_RETRIES = int(os.getenv("PLANNER_MAX_RETRIES", "3"))

# Engineer/Executor retry loop.
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "5"))
EXECUTOR_TIMEOUT_SECONDS = int(os.getenv("EXECUTOR_TIMEOUT_SECONDS", "600"))

# Paper ingestion pipeline.
INGEST_CLONE_TIMEOUT_SECONDS = int(os.getenv("INGEST_CLONE_TIMEOUT_SECONDS", "300"))
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://localhost/research_assistant")
