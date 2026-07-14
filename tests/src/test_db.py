"""Tests for the database layer."""

import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from src.db import DatabaseError, get_paper_by_id, init_schema, insert_paper, list_papers, paper_exists
from src.state import IngestedPaperRecord


@pytest.fixture
def test_db_dsn():
    """Create a temporary test database and return its DSN."""
    # Use a temporary database name
    test_db_name = f"test_research_assistant_{os.getpid()}"
    admin_dsn = "postgresql://localhost/postgres"
    test_dsn = f"postgresql://localhost/{test_db_name}"
    
    # Create test database
    conn = psycopg2.connect(admin_dsn)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute(f'CREATE DATABASE "{test_db_name}"')
    cursor.close()
    conn.close()
    
    yield test_dsn
    
    # Cleanup test database
    conn = psycopg2.connect(admin_dsn)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute(f'DROP DATABASE IF EXISTS "{test_db_name}"')
    cursor.close()
    conn.close()


@pytest.fixture
def mock_db_config(test_db_dsn):
    """Mock the database configuration to use test database."""
    with patch("src.db.POSTGRES_DSN", test_db_dsn):
        yield test_db_dsn


def test_init_schema(mock_db_config):
    """Test database schema initialization."""
    init_schema()
    
    # Verify table exists by attempting to query it
    with patch("src.db.POSTGRES_DSN", mock_db_config):
        from src.db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM papers")
                assert cursor.fetchone()[0] == 0


def test_insert_and_get_paper(mock_db_config):
    """Test inserting and retrieving a paper record."""
    init_schema()
    
    record = IngestedPaperRecord(
        paper_id="test_paper",
        title="Test Paper Title",
        arxiv_id="2024.001",
        repo_url="https://github.com/test/repo",
        bundle_path="/test/bundle/path",
        pdf_path="/test/bundle/path/paper.pdf",
        code_path="/test/bundle/path/code",
        pdf_checksum="abc123def456",
        ingested_at=datetime(2024, 1, 15, 10, 30, 0),
        repo_commit_sha="git123abc",
    )
    
    insert_paper(record)
    
    retrieved = get_paper_by_id("test_paper")
    assert retrieved is not None
    assert retrieved.paper_id == "test_paper"
    assert retrieved.title == "Test Paper Title"
    assert retrieved.arxiv_id == "2024.001"
    assert retrieved.repo_url == "https://github.com/test/repo"
    assert retrieved.bundle_path == "/test/bundle/path"
    assert retrieved.pdf_path == "/test/bundle/path/paper.pdf"
    assert retrieved.code_path == "/test/bundle/path/code"
    assert retrieved.pdf_checksum == "abc123def456"
    assert retrieved.repo_commit_sha == "git123abc"
    assert retrieved.ingested_at.year == 2024


def test_insert_duplicate_paper_id(mock_db_config):
    """Test that inserting duplicate paper_id raises DatabaseError."""
    init_schema()
    
    record1 = IngestedPaperRecord(
        paper_id="duplicate_paper",
        title="First Paper",
        bundle_path="/test/path1",
        pdf_path="/test/path1/paper.pdf",
        pdf_checksum="abc123",
        ingested_at=datetime.now(),
    )
    
    record2 = IngestedPaperRecord(
        paper_id="duplicate_paper",  # Same ID
        title="Second Paper",
        bundle_path="/test/path2",
        pdf_path="/test/path2/paper.pdf",
        pdf_checksum="def456",
        ingested_at=datetime.now(),
    )
    
    insert_paper(record1)
    
    with pytest.raises(DatabaseError, match="Paper 'duplicate_paper' already exists"):
        insert_paper(record2)


def test_get_nonexistent_paper(mock_db_config):
    """Test retrieving a non-existent paper returns None."""
    init_schema()
    
    result = get_paper_by_id("nonexistent_paper")
    assert result is None


def test_paper_exists(mock_db_config):
    """Test checking paper existence."""
    init_schema()
    
    assert not paper_exists("test_paper")
    
    record = IngestedPaperRecord(
        paper_id="test_paper",
        title="Test Paper",
        bundle_path="/test/path",
        pdf_path="/test/path/paper.pdf",
        pdf_checksum="abc123",
        ingested_at=datetime.now(),
    )
    
    insert_paper(record)
    
    assert paper_exists("test_paper")
    assert not paper_exists("other_paper")


def test_list_papers(mock_db_config):
    """Test listing all papers ordered by ingestion time."""
    init_schema()
    
    # Insert papers with different ingestion times
    record1 = IngestedPaperRecord(
        paper_id="paper1",
        title="First Paper",
        bundle_path="/test/path1",
        pdf_path="/test/path1/paper.pdf",
        pdf_checksum="abc123",
        ingested_at=datetime(2024, 1, 1, 10, 0, 0),
    )
    
    record2 = IngestedPaperRecord(
        paper_id="paper2",
        title="Second Paper",
        bundle_path="/test/path2",
        pdf_path="/test/path2/paper.pdf",
        pdf_checksum="def456",
        ingested_at=datetime(2024, 1, 2, 10, 0, 0),
    )
    
    insert_paper(record1)
    insert_paper(record2)
    
    papers = list_papers()
    assert len(papers) == 2
    
    # Should be ordered by ingestion time (newest first)
    assert papers[0].paper_id == "paper2"  # More recent
    assert papers[1].paper_id == "paper1"  # Older


def test_database_connection_error():
    """Test database error handling with invalid DSN."""
    with patch("src.db.POSTGRES_DSN", "postgresql://invalid:invalid@nonexistent:5432/invalid"):
        with pytest.raises(DatabaseError):
            init_schema()


def test_paper_with_minimal_fields(mock_db_config):
    """Test inserting and retrieving a paper with only required fields."""
    init_schema()
    
    record = IngestedPaperRecord(
        paper_id="minimal_paper",
        title="Minimal Paper",
        bundle_path="/test/minimal",
        pdf_path="/test/minimal/paper.pdf",
        pdf_checksum="minimal123",
        ingested_at=datetime.now(),
        # Optional fields left as None
        arxiv_id=None,
        repo_url=None,
        code_path=None,
        repo_commit_sha=None,
    )
    
    insert_paper(record)
    
    retrieved = get_paper_by_id("minimal_paper")
    assert retrieved is not None
    assert retrieved.paper_id == "minimal_paper"
    assert retrieved.title == "Minimal Paper"
    assert retrieved.arxiv_id is None
    assert retrieved.repo_url is None
    assert retrieved.code_path is None
    assert retrieved.repo_commit_sha is None