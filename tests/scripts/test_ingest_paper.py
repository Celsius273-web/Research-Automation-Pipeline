"""Tests for the paper ingestion script."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Mock database operations to avoid requiring PostgreSQL during tests
with patch("src.db.get_connection"), patch("src.db.init_schema"), patch("src.db.insert_paper"), patch("src.db.paper_exists", return_value=False):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    from ingest_paper import (
        clone_repository,
        compute_file_checksum,
        create_metadata_file,
        generate_unique_paper_id,
        ingest_paper,
        slugify,
        validate_clone,
        validate_pdf,
    )


def test_slugify():
    """Test PDF filename to paper_id conversion."""
    assert slugify("A Tutorial on Bayesian Optimization.pdf") == "a_tutorial_on_bayesian_optimization"
    assert slugify("Simple Title") == "simple_title"
    assert slugify("Title-with-dashes!@#") == "title_with_dashes"
    assert slugify("") == "paper"
    assert slugify("   ") == "paper"


def test_compute_file_checksum():
    """Test file checksum computation."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        test_file = Path(f.name)
    
    try:
        checksum = compute_file_checksum(test_file)
        assert len(checksum) == 64  # SHA-256 hex string length
        assert checksum == compute_file_checksum(test_file)  # Consistent
    finally:
        test_file.unlink()


def test_generate_unique_paper_id():
    """Test paper ID uniqueness generation."""
    with patch("scripts.ingest_paper.paper_exists") as mock_exists:
        mock_exists.return_value = False
        assert generate_unique_paper_id("test") == "test"
        
        mock_exists.side_effect = lambda x: x in ["test", "test_1"]
        assert generate_unique_paper_id("test") == "test_2"


def test_validate_pdf_success():
    """Test PDF validation with a valid file."""
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf") as f:
        f.write(b"%PDF-1.4\ntest content")  # Valid PDF header
        test_pdf = Path(f.name)
    
    try:
        validate_pdf(test_pdf)  # Should not raise
    finally:
        test_pdf.unlink()


def test_validate_pdf_missing_file():
    """Test PDF validation with missing file."""
    missing_file = Path("/nonexistent/file.pdf")
    with pytest.raises(ValueError, match="PDF file does not exist"):
        validate_pdf(missing_file)


def test_validate_pdf_invalid_format():
    """Test PDF validation with invalid file format."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pdf") as f:
        f.write("not a PDF file")
        test_file = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="does not appear to be a PDF"):
            validate_pdf(test_file)
    finally:
        test_file.unlink()


def test_validate_clone_success():
    """Test repository clone validation with valid structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        code_dir = Path(temp_dir) / "code"
        code_dir.mkdir()
        (code_dir / ".git").mkdir()
        (code_dir / "README.md").touch()
        
        validate_clone(code_dir)  # Should not raise


def test_validate_clone_missing_git():
    """Test repository clone validation without .git directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        code_dir = Path(temp_dir) / "code"
        code_dir.mkdir()
        
        with pytest.raises(ValueError, match=".git directory not found"):
            validate_clone(code_dir)


def test_validate_clone_empty_directory():
    """Test repository clone validation with empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        code_dir = Path(temp_dir) / "code"
        code_dir.mkdir()
        
        with pytest.raises(ValueError, match="appears to be empty"):
            validate_clone(code_dir)


def test_create_metadata_file():
    """Test metadata file creation."""
    from src.state import IngestedPaperRecord
    
    with tempfile.TemporaryDirectory() as temp_dir:
        bundle_dir = Path(temp_dir)
        record = IngestedPaperRecord(
            paper_id="test_paper",
            title="Test Paper",
            arxiv_id="2024.001",
            repo_url="https://github.com/test/repo",
            bundle_path=str(bundle_dir),
            pdf_path=str(bundle_dir / "paper.pdf"),
            code_path=str(bundle_dir / "code"),
            pdf_checksum="abc123",
            ingested_at=datetime.now(),
            repo_commit_sha="def456",
        )
        
        create_metadata_file(bundle_dir, record)
        
        metadata_file = bundle_dir / "metadata.json"
        assert metadata_file.exists()
        
        metadata = json.loads(metadata_file.read_text())
        assert metadata["paper_id"] == "test_paper"
        assert metadata["title"] == "Test Paper"
        assert metadata["arxiv_id"] == "2024.001"
        assert metadata["repo_url"] == "https://github.com/test/repo"
        assert metadata["pdf_checksum"] == "abc123"
        assert metadata["repo_commit_sha"] == "def456"
        assert "bundle_structure" in metadata


@patch("scripts.ingest_paper.insert_paper")
@patch("scripts.ingest_paper.paper_exists", return_value=False)
@patch("scripts.ingest_paper.PAPER_BUNDLES_DIR")
def test_ingest_paper_pdf_only(mock_bundles_dir, mock_exists, mock_insert):
    """Test paper ingestion with PDF only (no repository)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_bundles_dir.__truediv__.side_effect = lambda x: Path(temp_dir) / x
        
        # Create test PDF
        test_pdf = Path(temp_dir) / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4\ntest content")
        
        record = ingest_paper(
            pdf_path=str(test_pdf),
            paper_id="test_paper",
            title="Test Paper"
        )
        
        assert record.paper_id == "test_paper"
        assert record.title == "Test Paper"
        assert record.repo_url is None
        assert record.code_path is None
        assert record.repo_commit_sha is None
        assert len(record.pdf_checksum) == 64
        
        mock_insert.assert_called_once()


@patch("scripts.ingest_paper.insert_paper")
@patch("scripts.ingest_paper.paper_exists", return_value=False)
@patch("scripts.ingest_paper.clone_repository", return_value="abc123")
@patch("scripts.ingest_paper.validate_clone")
@patch("scripts.ingest_paper.PAPER_BUNDLES_DIR")
def test_ingest_paper_with_repo(mock_bundles_dir, mock_validate, mock_clone, mock_exists, mock_insert):
    """Test paper ingestion with PDF and repository."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_bundles_dir.__truediv__.side_effect = lambda x: Path(temp_dir) / x
        
        # Create test PDF
        test_pdf = Path(temp_dir) / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4\ntest content")
        
        record = ingest_paper(
            pdf_path=str(test_pdf),
            repo_url="https://github.com/test/repo",
            paper_id="test_paper",
            title="Test Paper"
        )
        
        assert record.paper_id == "test_paper"
        assert record.title == "Test Paper"
        assert record.repo_url == "https://github.com/test/repo"
        assert record.code_path is not None
        assert record.repo_commit_sha == "abc123"
        
        mock_clone.assert_called_once()
        mock_validate.assert_called_once()
        mock_insert.assert_called_once()


@patch("scripts.ingest_paper.paper_exists", return_value=True)
def test_ingest_paper_duplicate_id(mock_exists):
    """Test paper ingestion failure when paper_id already exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_pdf = Path(temp_dir) / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4\ntest content")
        
        with pytest.raises(ValueError, match="Paper ID 'test_paper' already exists"):
            ingest_paper(
                pdf_path=str(test_pdf),
                paper_id="test_paper"
            )


@patch("scripts.ingest_paper.insert_paper")
@patch("scripts.ingest_paper.paper_exists", return_value=False)
@patch("scripts.ingest_paper.clone_repository", side_effect=ValueError("Clone failed"))
@patch("scripts.ingest_paper.PAPER_BUNDLES_DIR")
def test_ingest_paper_clone_failure_cleanup(mock_bundles_dir, mock_clone, mock_exists, mock_insert):
    """Test that bundle directory is cleaned up when repository clone fails."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_bundles_dir.__truediv__.side_effect = lambda x: Path(temp_dir) / x
        
        # Create test PDF
        test_pdf = Path(temp_dir) / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4\ntest content")
        
        with pytest.raises(ValueError, match="Clone failed"):
            ingest_paper(
                pdf_path=str(test_pdf),
                repo_url="https://github.com/test/repo",
                paper_id="test_paper"
            )
        
        # Bundle directory should be cleaned up
        bundle_path = Path(temp_dir) / "test_paper"
        assert not bundle_path.exists()
        
        # Database insertion should not have been attempted
        mock_insert.assert_not_called()