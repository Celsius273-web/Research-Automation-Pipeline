#!/usr/bin/env python3
"""Paper ingestion script - the sole entry point for adding papers to the research pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import INGEST_CLONE_TIMEOUT_SECONDS, PAPER_BUNDLES_DIR
from src.db import DatabaseError, get_paper_by_id, init_schema, insert_paper, paper_exists
from src.state import IngestedPaperRecord


def slugify(value: str) -> str:
    """Convert a string to a filesystem-safe identifier."""
    base = value.lower().strip()
    base = re.sub(r"\.pdf$", "", base)
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return base or "paper"


def compute_file_checksum(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def generate_unique_paper_id(base_id: str) -> str:
    """Generate a unique paper_id by appending suffix if needed."""
    if not paper_exists(base_id):
        return base_id
    
    counter = 1
    while True:
        candidate = f"{base_id}_{counter}"
        if not paper_exists(candidate):
            return candidate
        counter += 1


def validate_pdf(pdf_path: Path) -> None:
    """Validate that the PDF exists and is readable."""
    if not pdf_path.exists():
        raise ValueError(f"PDF file does not exist: {pdf_path}")
    
    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")
    
    # Basic PDF validation - check file header
    try:
        with pdf_path.open("rb") as f:
            header = f.read(5)
            if not header.startswith(b"%PDF-"):
                raise ValueError(f"File does not appear to be a PDF: {pdf_path}")
    except Exception as e:
        raise ValueError(f"Cannot read PDF file {pdf_path}: {e}") from e


def clone_repository(repo_url: str, target_dir: Path) -> str:
    """Clone repository to target directory. Returns the commit SHA."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Clone with shallow depth for efficiency
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
            capture_output=True,
            text=True,
            timeout=INGEST_CLONE_TIMEOUT_SECONDS,
            check=True,
        )
        
        # Get the commit SHA
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        
        commit_sha = sha_result.stdout.strip()
        print(f"Successfully cloned repository to {target_dir}")
        print(f"Repository commit SHA: {commit_sha}")
        
        return commit_sha
        
    except subprocess.TimeoutExpired as e:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        raise ValueError(f"Repository clone timed out after {INGEST_CLONE_TIMEOUT_SECONDS}s: {repo_url}") from e
    except subprocess.CalledProcessError as e:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        raise ValueError(f"Failed to clone repository {repo_url}: {e.stderr}") from e


def validate_clone(code_dir: Path) -> None:
    """Validate that the repository was cloned successfully."""
    git_dir = code_dir / ".git"
    if not git_dir.exists():
        raise ValueError(f"Clone validation failed: .git directory not found in {code_dir}")
    
    # Check that the working tree is readable
    if not any(code_dir.iterdir()):
        raise ValueError(f"Clone validation failed: {code_dir} appears to be empty")


def create_metadata_file(bundle_dir: Path, record: IngestedPaperRecord) -> None:
    """Create metadata.json file in the bundle directory."""
    metadata = {
        "paper_id": record.paper_id,
        "title": record.title,
        "arxiv_id": record.arxiv_id,
        "repo_url": record.repo_url,
        "pdf_checksum": record.pdf_checksum,
        "ingested_at": record.ingested_at.isoformat(),
        "repo_commit_sha": record.repo_commit_sha,
        "bundle_structure": {
            "paper.pdf": "Original paper PDF",
            "code/": "Cloned repository (if repo_url provided)",
            "metadata.json": "This metadata file"
        }
    }
    
    metadata_path = bundle_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def ingest_paper(
    pdf_path: str,
    repo_url: str | None = None,
    paper_id: str | None = None,
    title: str | None = None,
    arxiv_id: str | None = None,
) -> IngestedPaperRecord:
    """Main ingestion logic."""
    
    # Validate input PDF
    source_pdf = Path(pdf_path).resolve()
    validate_pdf(source_pdf)
    
    # Generate paper ID
    if paper_id is None:
        base_id = slugify(source_pdf.stem)
        paper_id = generate_unique_paper_id(base_id)
    else:
        if paper_exists(paper_id):
            raise ValueError(f"Paper ID '{paper_id}' already exists in database")
    
    # Create bundle directory
    bundle_dir = PAPER_BUNDLES_DIR / paper_id
    if bundle_dir.exists():
        raise ValueError(f"Bundle directory already exists: {bundle_dir}")
    
    bundle_dir.mkdir(parents=True)
    print(f"Created bundle directory: {bundle_dir}")
    
    try:
        # Copy PDF
        pdf_dest = bundle_dir / "paper.pdf"
        shutil.copy2(source_pdf, pdf_dest)
        print(f"Copied PDF: {source_pdf} -> {pdf_dest}")
        
        # Compute PDF checksum
        pdf_checksum = compute_file_checksum(pdf_dest)
        
        # Handle repository cloning
        code_path = None
        commit_sha = None
        if repo_url:
            code_dir = bundle_dir / "code"
            commit_sha = clone_repository(repo_url, code_dir)
            validate_clone(code_dir)
            code_path = str(code_dir)
        
        # Create ingested paper record
        record = IngestedPaperRecord(
            paper_id=paper_id,
            title=title or source_pdf.stem,
            arxiv_id=arxiv_id,
            repo_url=repo_url,
            bundle_path=str(bundle_dir),
            pdf_path=str(pdf_dest),
            code_path=code_path,
            pdf_checksum=pdf_checksum,
            ingested_at=datetime.now(),
            repo_commit_sha=commit_sha,
        )
        
        # Create metadata file
        create_metadata_file(bundle_dir, record)
        print(f"Created metadata file: {bundle_dir / 'metadata.json'}")
        
        # Register in database
        insert_paper(record)
        print(f"Registered paper in database: {paper_id}")
        
        return record
        
    except Exception:
        # Cleanup on failure
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
            print(f"Cleaned up bundle directory due to failure: {bundle_dir}")
        raise


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Ingest a research paper (PDF + optional repository) into the research pipeline"
    )
    parser.add_argument("--pdf-path", required=True, help="Path to the paper PDF file")
    parser.add_argument("--repo-url", help="Optional GitHub repository URL to clone")
    parser.add_argument("--paper-id", help="Optional custom paper ID (auto-generated if not provided)")
    parser.add_argument("--title", help="Optional paper title (defaults to PDF filename)")
    parser.add_argument("--arxiv-id", help="Optional arXiv ID")
    parser.add_argument("--init-db", action="store_true", help="Initialize database schema before ingestion")
    
    args = parser.parse_args()
    
    try:
        if args.init_db:
            print("Initializing database schema...")
            init_schema()
            print("Database schema initialized successfully")
        
        record = ingest_paper(
            pdf_path=args.pdf_path,
            repo_url=args.repo_url,
            paper_id=args.paper_id,
            title=args.title,
            arxiv_id=args.arxiv_id,
        )
        
        print("\n=== INGESTION SUCCESSFUL ===")
        print(f"Paper ID: {record.paper_id}")
        print(f"Bundle Path: {record.bundle_path}")
        print(f"PDF Path: {record.pdf_path}")
        if record.code_path:
            print(f"Code Path: {record.code_path}")
            print(f"Repository Commit: {record.repo_commit_sha}")
        print(f"Ingested At: {record.ingested_at}")
        
        return 0
        
    except ValueError as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        return 1
    except DatabaseError as e:
        print(f"Database Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())