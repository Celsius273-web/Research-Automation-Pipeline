"""Tests for main.py ingestion enforcement."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.main import run_analyze, run_execute, run_review
from src.state import IngestedPaperRecord, PaperMetadata, ReviewerReport, empty_planner_envelope


@patch("src.main.get_paper_by_id")
def test_run_analyze_paper_not_found(mock_get_paper):
    """Test that analyze refuses unknown paper IDs."""
    mock_get_paper.return_value = None
    
    result = run_analyze(
        paper_id="unknown_paper",
        non_interactive=True,
        with_plan=False
    )
    
    assert result == 1
    mock_get_paper.assert_called_once_with("unknown_paper")


@patch("src.main.get_paper_by_id")
def test_run_analyze_database_error(mock_get_paper):
    """Test that analyze handles database errors gracefully."""
    from src.db import DatabaseError
    mock_get_paper.side_effect = DatabaseError("Connection failed")
    
    result = run_analyze(
        paper_id="test_paper",
        non_interactive=True,
        with_plan=False
    )
    
    assert result == 1


@patch("src.main.get_paper_by_id")
def test_run_analyze_missing_pdf_file(mock_get_paper):
    """Test that analyze fails when ingested PDF file is missing."""
    mock_get_paper.return_value = IngestedPaperRecord(
        paper_id="test_paper",
        title="Test Paper",
        bundle_path="/nonexistent/bundle",
        pdf_path="/nonexistent/bundle/paper.pdf",
        pdf_checksum="abc123",
        ingested_at=datetime.now(),
    )
    
    result = run_analyze(
        paper_id="test_paper",
        non_interactive=True,
        with_plan=False
    )
    
    assert result == 1


@patch("src.main.get_paper_by_id")
@patch("src.main.make_phase1_nodes")
@patch("src.main.build_phase1_graph")
@patch("src.main.persist_extraction")
def test_run_analyze_success(mock_persist, mock_build_graph, mock_make_nodes, mock_get_paper):
    """Test successful analyze with ingested paper."""
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\ntest content")
        
        mock_get_paper.return_value = IngestedPaperRecord(
            paper_id="test_paper",
            title="Test Paper",
            arxiv_id="2024.001",
            bundle_path=temp_dir,
            pdf_path=str(pdf_path),
            pdf_checksum="abc123",
            ingested_at=datetime.now(),
        )
        
        # Mock the graph execution
        mock_graph = Mock()
        mock_graph.invoke.return_value = {
            "paper": PaperMetadata(
                paper_id="test_paper",
                title="Test Paper",
                pdf_path=str(pdf_path),
                arxiv_id="2024.001"
            ),
            "approved_extraction": Mock(),
            "review": Mock(status="approved"),
            "errors": []
        }
        mock_build_graph.return_value = mock_graph
        mock_make_nodes.return_value = (Mock(), Mock(), Mock())
        mock_persist.return_value = Path("/test/extraction.json")
        
        result = run_analyze(
            paper_id="test_paper",
            non_interactive=True,
            with_plan=False
        )
        
        assert result == 0
        mock_get_paper.assert_called_once_with("test_paper")
        mock_persist.assert_called_once()


@patch("src.main.get_paper_by_id")
@patch("src.main.make_phase1_nodes")
@patch("src.main.build_phase1_graph")
@patch("src.main.persist_extraction")
def test_run_analyze_fails_fast_when_graph_has_errors(
    mock_persist,
    mock_build_graph,
    mock_make_nodes,
    mock_get_paper,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\ntest content")
        mock_get_paper.return_value = IngestedPaperRecord(
            paper_id="test_paper",
            title="Test Paper",
            arxiv_id="2024.001",
            bundle_path=temp_dir,
            pdf_path=str(pdf_path),
            pdf_checksum="abc123",
            ingested_at=datetime.now(),
        )
        mock_graph = Mock()
        mock_graph.invoke.return_value = {
            "paper": PaperMetadata(
                paper_id="test_paper",
                title="Test Paper",
                pdf_path=str(pdf_path),
                arxiv_id="2024.001",
            ),
            "approved_extraction": Mock(),
            "review": Mock(status="approved"),
            "errors": ["Planner failed to return valid JSON"],
        }
        mock_build_graph.return_value = mock_graph
        mock_make_nodes.return_value = (Mock(), Mock(), Mock())

        result = run_analyze(paper_id="test_paper", non_interactive=True, with_plan=False)

        assert result == 1
        mock_persist.assert_not_called()


@patch("src.main.get_paper_by_id")
@patch("src.main.make_phase1_nodes")
@patch("src.main.make_planner_node")
@patch("src.main.build_phase2_graph")
@patch("src.main.persist_extraction")
@patch("src.main.persist_plan")
def test_run_analyze_does_not_persist_default_plan_output(
    mock_persist_plan,
    mock_persist_extraction,
    mock_build_graph,
    mock_make_planner_node,
    mock_make_nodes,
    mock_get_paper,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\ntest content")
        mock_get_paper.return_value = IngestedPaperRecord(
            paper_id="test_paper",
            title="Test Paper",
            arxiv_id="2024.001",
            bundle_path=temp_dir,
            pdf_path=str(pdf_path),
            pdf_checksum="abc123",
            ingested_at=datetime.now(),
        )
        mock_graph = Mock()
        mock_graph.invoke.return_value = {
            "paper": PaperMetadata(
                paper_id="test_paper",
                title="Test Paper",
                pdf_path=str(pdf_path),
                arxiv_id="2024.001",
            ),
            "approved_extraction": Mock(),
            "review": Mock(status="approved"),
            "planner_output": empty_planner_envelope(),
            "plan_review": Mock(status="approved"),
            "errors": [],
        }
        mock_build_graph.return_value = mock_graph
        mock_make_nodes.return_value = (Mock(), Mock(), Mock())
        mock_make_planner_node.return_value = Mock()
        mock_persist_extraction.return_value = Path("/test/extraction.json")

        result = run_analyze(paper_id="test_paper", non_interactive=True, with_plan=True)

        assert result == 0
        mock_persist_extraction.assert_called_once()
        mock_persist_plan.assert_not_called()


@patch("src.main.get_paper_by_id")
@patch("src.main.resolve_plan_path")
def test_run_execute_uses_ingested_repo(mock_resolve_plan, mock_get_paper):
    """Test that execute uses ingested repository when no repo_path provided."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plan_file = Path(temp_dir) / "plan.json"
        code_dir = Path(temp_dir) / "code"
        code_dir.mkdir()
        
        # Create mock plan file
        plan_data = {
            "paper": {
                "paper_id": "test_paper",
                "title": "Test Paper",
                "pdf_path": "/test/paper.pdf"
            },
            "execution_plan": {
                "schema_version": "1.0",
                "plan_summary": "Test plan",
                "steps": []
            },
            "source_extraction_path": "/test/extraction.json"
        }
        plan_file.write_text(json.dumps(plan_data))
        mock_resolve_plan.return_value = plan_file
        
        mock_get_paper.return_value = IngestedPaperRecord(
            paper_id="test_paper",
            title="Test Paper",
            bundle_path=temp_dir,
            pdf_path="/test/paper.pdf",
            code_path=str(code_dir),
            pdf_checksum="abc123",
            ingested_at=datetime.now(),
            repo_commit_sha="abc123"
        )
        
        with patch("src.main.detect_language") as mock_detect, \
             patch("src.main.make_engineer_executor_nodes") as mock_make_nodes, \
             patch("src.main.build_phase3_graph") as mock_build_graph, \
             patch("src.main.persist_run_summary") as mock_persist:
            
            mock_detect.return_value = Mock()
            mock_make_nodes.return_value = (Mock(), Mock(), Mock())
            mock_graph = Mock()
            mock_graph.invoke.return_value = {
                "paper": Mock(),
                "planner_output": Mock(),
                "repo_context": Mock(),
                "executor_result": Mock(final_status="success"),
                "errors": []
            }
            mock_build_graph.return_value = mock_graph
            mock_persist.return_value = Path("/test/run_summary.json")
            
            result = run_execute(
                plan_path=None,
                paper_id="test_paper",
                repo_path=None,  # No explicit repo path
                non_interactive=True,
                with_review=False
            )
            
            assert result == 0
            mock_get_paper.assert_called_once_with("test_paper")
            mock_detect.assert_called_once_with(repo_path=str(code_dir))


@patch("src.main.get_paper_by_id")
@patch("src.main.resolve_plan_path")
def test_run_execute_no_ingested_repo(mock_resolve_plan, mock_get_paper):
    """Test that execute fails when no repo_path provided and no ingested repo."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plan_file = Path(temp_dir) / "plan.json"
        
        plan_data = {
            "paper": {
                "paper_id": "test_paper",
                "title": "Test Paper",
                "pdf_path": "/test/paper.pdf"
            },
            "execution_plan": {
                "schema_version": "1.0",
                "plan_summary": "Test plan",
                "steps": []
            }
        }
        plan_file.write_text(json.dumps(plan_data))
        mock_resolve_plan.return_value = plan_file
        
        # No code_path in ingested record
        mock_get_paper.return_value = IngestedPaperRecord(
            paper_id="test_paper",
            title="Test Paper",
            bundle_path=temp_dir,
            pdf_path="/test/paper.pdf",
            code_path=None,  # No ingested repository
            pdf_checksum="abc123",
            ingested_at=datetime.now(),
        )
        
        result = run_execute(
            plan_path=None,
            paper_id="test_paper",
            repo_path=None,
            non_interactive=True,
            with_review=False
        )
        
        assert result == 1


@patch("src.main.resolve_plan_path")
def test_run_execute_explicit_repo_path_takes_precedence(mock_resolve_plan):
    """Test that explicit repo_path is used even when ingested repo exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plan_file = Path(temp_dir) / "plan.json"
        explicit_repo = Path(temp_dir) / "explicit_repo"
        explicit_repo.mkdir()
        
        plan_data = {
            "paper": {
                "paper_id": "test_paper",
                "title": "Test Paper",
                "pdf_path": "/test/paper.pdf"
            },
            "execution_plan": {
                "schema_version": "1.0",
                "plan_summary": "Test plan",
                "steps": []
            }
        }
        plan_file.write_text(json.dumps(plan_data))
        mock_resolve_plan.return_value = plan_file
        
        with patch("src.main.detect_language") as mock_detect, \
             patch("src.main.make_engineer_executor_nodes") as mock_make_nodes, \
             patch("src.main.build_phase3_graph") as mock_build_graph, \
             patch("src.main.persist_run_summary") as mock_persist:
            
            mock_detect.return_value = Mock()
            mock_make_nodes.return_value = (Mock(), Mock(), Mock())
            mock_graph = Mock()
            mock_graph.invoke.return_value = {
                "paper": Mock(),
                "planner_output": Mock(),
                "repo_context": Mock(),
                "executor_result": Mock(final_status="success"),
                "errors": []
            }
            mock_build_graph.return_value = mock_graph
            mock_persist.return_value = Path("/test/run_summary.json")
            
            result = run_execute(
                plan_path=None,
                paper_id="test_paper",
                repo_path=str(explicit_repo),  # Explicit path provided
                non_interactive=True,
                with_review=False
            )
            
            assert result == 0
            # Should use explicit repo path, not look up ingested paper
            mock_detect.assert_called_once_with(repo_path=str(explicit_repo))


@patch("src.main.resolve_plan_path")
def test_run_execute_returns_failure_when_graph_sets_errors(mock_resolve_plan):
    with tempfile.TemporaryDirectory() as temp_dir:
        plan_file = Path(temp_dir) / "plan.json"
        explicit_repo = Path(temp_dir) / "repo"
        explicit_repo.mkdir()
        plan_data = {
            "paper": {
                "paper_id": "test_paper",
                "title": "Test Paper",
                "pdf_path": "/test/paper.pdf",
            },
            "execution_plan": {
                "schema_version": "1.0",
                "plan_summary": "Test plan",
                "steps": [],
            },
        }
        plan_file.write_text(json.dumps(plan_data))
        mock_resolve_plan.return_value = plan_file

        with (
            patch("src.main.detect_language") as mock_detect,
            patch("src.main.make_engineer_executor_nodes") as mock_make_nodes,
            patch("src.main.build_phase3_graph") as mock_build_graph,
            patch("src.main.persist_run_summary") as mock_persist,
        ):
            mock_detect.return_value = Mock()
            mock_make_nodes.return_value = (Mock(), Mock(), Mock())
            mock_graph = Mock()
            mock_graph.invoke.return_value = {
                "errors": ["Engineer failed to produce patch"],
                "executor_result": Mock(final_status="failed"),
            }
            mock_build_graph.return_value = mock_graph

            result = run_execute(
                plan_path=None,
                paper_id="test_paper",
                repo_path=str(explicit_repo),
                non_interactive=True,
                with_review=False,
            )

            assert result == 1
            mock_persist.assert_not_called()


@patch("src.main.resolve_run_summary_path")
@patch("src.main.make_reviewer_node")
@patch("src.main.persist_report")
def test_run_review_returns_failure_when_reviewer_sets_errors(
    mock_persist_report,
    mock_make_reviewer_node,
    mock_resolve_run_summary_path,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        summary_path = Path(temp_dir) / "run_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "paper": {"paper_id": "test_paper", "title": "Test Paper", "pdf_path": "/tmp/paper.pdf"},
                    "execution_plan": {"schema_version": "1.0", "steps": []},
                    "repo_context": {"repo_path": "/tmp/repo", "language": "python", "build_system": "pytest"},
                    "executor_result": {"attempts": [], "captured_metrics": [], "final_status": "success"},
                }
            ),
            encoding="utf-8",
        )
        mock_resolve_run_summary_path.return_value = summary_path
        reviewer_node = Mock(return_value={"errors": ["Reviewer model unavailable"], "paper": Mock()})
        mock_make_reviewer_node.return_value = reviewer_node

        result = run_review(run_path=None, paper_id="test_paper", extraction_path=None)

        assert result == 1
        mock_persist_report.assert_not_called()