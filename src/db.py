"""PostgreSQL persistence layer for paper ingestion registry."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

import psycopg2
from psycopg2.extras import RealDictCursor

from src.config import POSTGRES_DSN
from src.state import IngestedPaperRecord

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Get database connection with automatic cleanup."""
    conn = None
    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        yield conn
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        raise DatabaseError(f"Database operation failed: {e}") from e
    finally:
        if conn:
            conn.close()


def init_schema() -> None:
    """Initialize the papers table schema."""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS papers (
        paper_id VARCHAR(255) PRIMARY KEY,
        title TEXT NOT NULL,
        arxiv_id VARCHAR(50),
        repo_url TEXT,
        bundle_path TEXT NOT NULL,
        pdf_path TEXT NOT NULL,
        code_path TEXT,
        pdf_checksum VARCHAR(64) NOT NULL,
        ingested_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        repo_commit_sha VARCHAR(40)
    );

    CREATE INDEX IF NOT EXISTS idx_papers_ingested_at ON papers(ingested_at);
    CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers(arxiv_id) WHERE arxiv_id IS NOT NULL;
    """
    
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(schema_sql)
            conn.commit()
            logger.info("Papers table schema initialized")


def insert_paper(record: IngestedPaperRecord) -> None:
    """Insert a new paper record. Raises DatabaseError if paper_id already exists."""
    insert_sql = """
    INSERT INTO papers (
        paper_id, title, arxiv_id, repo_url, bundle_path, pdf_path, 
        code_path, pdf_checksum, ingested_at, repo_commit_sha
    ) VALUES (
        %(paper_id)s, %(title)s, %(arxiv_id)s, %(repo_url)s, %(bundle_path)s, 
        %(pdf_path)s, %(code_path)s, %(pdf_checksum)s, %(ingested_at)s, %(repo_commit_sha)s
    )
    """
    
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(insert_sql, record.model_dump())
                conn.commit()
                logger.info(f"Inserted paper record: {record.paper_id}")
            except psycopg2.IntegrityError as e:
                raise DatabaseError(f"Paper '{record.paper_id}' already exists") from e


def get_paper_by_id(paper_id: str) -> IngestedPaperRecord | None:
    """Retrieve paper record by paper_id. Returns None if not found."""
    select_sql = "SELECT * FROM papers WHERE paper_id = %s"
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_sql, (paper_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
                
            return IngestedPaperRecord(**dict(row))


def list_papers() -> list[IngestedPaperRecord]:
    """List all ingested papers ordered by ingestion time (newest first)."""
    select_sql = "SELECT * FROM papers ORDER BY ingested_at DESC"
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_sql)
            rows = cursor.fetchall()
            
            return [IngestedPaperRecord(**dict(row)) for row in rows]


def paper_exists(paper_id: str) -> bool:
    """Check if a paper with the given ID already exists."""
    select_sql = "SELECT 1 FROM papers WHERE paper_id = %s LIMIT 1"
    
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(select_sql, (paper_id,))
            return cursor.fetchone() is not None