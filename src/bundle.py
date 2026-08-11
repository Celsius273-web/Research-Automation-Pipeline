"""Paper bundle management system for colocated artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.config import PAPER_BUNDLES_DIR, PLANNER_DEFAULT_SETUP_MINUTES
from src.state import (
    AgentEnvelope,
    ExtractionBundle,
    PaperMetadata,
    PlannerPayload,
    RepoContext,
    ReviewerReport,
    ReviewRecord,
    SectionExtraction,
)
from src.tools.language_detect import detect_language
from src.tools.repo_context import (
    extract_entrypoint_hints,
    extract_example_commands,
    infer_build_command,
    summarize_readme,
    summarize_repo_tree,
)

logger = logging.getLogger(__name__)


class PaperBundle:
    """Manages all artifacts for a single paper in a colocated directory structure."""

    def __init__(self, paper_id: str):
        self.paper_id = paper_id
        self.bundle_dir = PAPER_BUNDLES_DIR / paper_id
        self.metadata_path = self.bundle_dir / "metadata.json"
        self.pdf_path = self.bundle_dir / "paper.pdf"
        self.code_dir = self.bundle_dir / "code"
        self.extraction_path = self.bundle_dir / f"{paper_id}.json"
        self.plan_path = self.bundle_dir / f"{paper_id}_plan.json"
        self.runs_dir = self.bundle_dir / "runs"
        self.report_path = self.bundle_dir / "report.json"

    def exists(self) -> bool:
        """Check if the bundle directory exists."""
        return self.bundle_dir.exists()

    def create_bundle_dir(self) -> None:
        """Create the bundle directory structure."""
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)

    def get_metadata(self) -> Optional[dict]:
        """Load paper metadata from bundle."""
        if not self.metadata_path.exists():
            return None
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            logger.warning("Failed to load metadata for %s: %s", self.paper_id, exc)
            return None

    def save_metadata(self, metadata: dict) -> None:
        """Save paper metadata to bundle."""
        self.create_bundle_dir()
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def has_code(self) -> bool:
        """Check if bundle contains cloned repository code."""
        return self.code_dir.exists() and any(self.code_dir.iterdir())

    def get_repo_info(self) -> RepoContext:
        """Get repository context information."""
        metadata = self.get_metadata() or {}
        repo_url = str(metadata.get("repo_url") or "")
        if not self.has_code():
            return RepoContext(
                repo_url=repo_url,
                repo_path=str(self.code_dir),
                language="unknown",
                build_system="unknown",
                has_code=False,
                notes="No code repository available",
            )

        detected = detect_language(repo_path=str(self.code_dir), repo_url=repo_url)
        detected.has_code = True
        detected.setup_time_minutes = PLANNER_DEFAULT_SETUP_MINUTES
        detected.file_tree = summarize_repo_tree(self.code_dir)
        detected.readme_summary = summarize_readme(self.code_dir)
        detected.build_system = infer_build_command(self.code_dir, detected.build_system)
        detected.example_commands = extract_example_commands(self.code_dir)
        detected.entrypoint_hints = extract_entrypoint_hints(self.code_dir)
        return detected

    def get_setup_guide(self) -> str:
        """Extract setup instructions from README files in the code directory."""
        if not self.has_code():
            return ""

        # Look for common README files
        readme_patterns = ["README.md", "README.txt", "README.rst", "README", "INSTALL.md", "INSTALL.txt"]
        
        for pattern in readme_patterns:
            readme_path = self.code_dir / pattern
            if readme_path.exists():
                try:
                    content = readme_path.read_text(encoding="utf-8")
                    # Extract installation/setup sections (simple heuristic)
                    lines = content.split('\n')
                    setup_lines = []
                    in_setup_section = False
                    
                    for line in lines:
                        lower_line = line.lower()
                        if any(keyword in lower_line for keyword in ['install', 'setup', 'requirements', 'getting started']):
                            in_setup_section = True
                        elif line.startswith('#') and in_setup_section and setup_lines:
                            # New section, stop if we already collected some setup info
                            break
                        
                        if in_setup_section:
                            setup_lines.append(line)
                    
                    if setup_lines:
                        return '\n'.join(setup_lines[:20])  # Limit to first 20 lines
                    else:
                        return content[:2000]  # First 2000 chars as fallback
                        
                except (UnicodeDecodeError, FileNotFoundError) as exc:
                    logger.warning("Failed to read %s: %s", readme_path, exc)
                    continue

        return ""

    def get_extraction(self) -> Optional[ExtractionBundle]:
        """Load extraction bundle from the paper bundle."""
        if not self.extraction_path.exists():
            return None
        
        try:
            data = json.loads(self.extraction_path.read_text(encoding="utf-8"))
            # Handle both new bundle format and legacy format
            if "by_section" in data and "merged" in data:
                return ExtractionBundle.model_validate(data)
            elif "approved_extraction" in data:
                # Legacy format - convert to bundle
                merged = SectionExtraction.model_validate(data["approved_extraction"])
                return ExtractionBundle(by_section={}, merged=merged)
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as exc:
            logger.warning("Failed to load extraction for %s: %s", self.paper_id, exc)
        
        return None

    def save_extraction(self, bundle: ExtractionBundle, review: ReviewRecord, paper: PaperMetadata) -> None:
        """Save extraction bundle to the paper bundle."""
        self.create_bundle_dir()
        
        payload = {
            "paper": paper.model_dump(),
            "review": review.model_dump(),
            "by_section": {k: v.model_dump() for k, v in bundle.by_section.items()},
            "merged": bundle.merged.model_dump(),
        }
        
        self.extraction_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_plan(self) -> Optional[AgentEnvelope[PlannerPayload]]:
        """Load execution plan from the paper bundle."""
        if not self.plan_path.exists():
            return None
            
        try:
            from src.persistence import load_planner_envelope

            data = json.loads(self.plan_path.read_text(encoding="utf-8"))
            return load_planner_envelope(data)
                
        except (json.JSONDecodeError, FileNotFoundError, KeyError, ValueError) as exc:
            logger.warning("Failed to load plan for %s: %s", self.paper_id, exc)
        
        return None

    def save_plan(
        self,
        plan: AgentEnvelope[PlannerPayload],
        plan_review: ReviewRecord,
        paper: PaperMetadata,
        source_extraction_path: Optional[str] = None,
    ) -> None:
        """Save execution plan to the paper bundle."""
        self.create_bundle_dir()
        
        payload = {
            "paper": paper.model_dump(),
            "plan_review": plan_review.model_dump(),
            "source_extraction_path": source_extraction_path or str(self.extraction_path),
            "plan_envelope": plan.model_dump(),
        }
        
        self.plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_report(self) -> Optional[ReviewerReport]:
        """Load reviewer report from the paper bundle."""
        if not self.report_path.exists():
            return None
            
        try:
            data = json.loads(self.report_path.read_text(encoding="utf-8"))
            return ReviewerReport.model_validate(data)
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            logger.warning("Failed to load report for %s: %s", self.paper_id, exc)
        
        return None

    def save_report(self, report: ReviewerReport, paper: PaperMetadata) -> None:
        """Save reviewer report to the paper bundle."""
        self.create_bundle_dir()
        
        payload = {
            "paper": paper.model_dump(),
            "reviewer_report": report.model_dump(),
        }
        
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_dependency_files(self) -> list[str]:
        """Find all dependency configuration files (setup.py, requirements.txt, pyproject.toml)."""
        if not self.has_code():
            return []
        
        dependency_files = []
        patterns = ["setup.py", "requirements.txt", "pyproject.toml", "setup.cfg", "Pipfile"]
        
        # Root level files
        for pattern in patterns:
            path = self.code_dir / pattern
            if path.exists():
                dependency_files.append(str(path.relative_to(self.code_dir)))
        
        # Subdirectory files (e.g., HEBO/requirements.txt)
        for subdir in self.code_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                for pattern in patterns:
                    path = subdir / pattern
                    if path.exists():
                        dependency_files.append(str(path.relative_to(self.code_dir)))
        
        return sorted(list(set(dependency_files)))

    def get_hyperparameter_reference(self) -> str:
        """Extract hyperparameter tables from the extraction."""
        extraction = self.get_extraction()
        if not extraction:
            return ""
            
        # Combine hyperparameters from merged extraction
        merged = extraction.merged
        if not merged.hyperparameters:
            return ""
            
        # Format as reference string
        ref_lines = ["Hyperparameter Reference:"]
        for key, value in merged.hyperparameters.items():
            ref_lines.append(f"- {key}: {value}")
            
        return "\n".join(ref_lines)

    def migrate_from_legacy(
        self,
        extraction_path: Optional[Path] = None,
        plan_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
    ) -> None:
        """Migrate artifacts from legacy directory structure to bundle."""
        self.create_bundle_dir()
        
        # Migrate extraction
        if extraction_path and extraction_path.exists():
            try:
                data = json.loads(extraction_path.read_text(encoding="utf-8"))
                if "by_section" in data and "merged" in data:
                    # Already in bundle format
                    self.extraction_path.write_text(extraction_path.read_text(encoding="utf-8"), encoding="utf-8")
                elif "approved_extraction" in data:
                    # Convert legacy format
                    paper = PaperMetadata.model_validate(data["paper"])
                    review = ReviewRecord.model_validate(data["review"])
                    merged = SectionExtraction.model_validate(data["approved_extraction"])
                    bundle = ExtractionBundle(by_section={}, merged=merged)
                    self.save_extraction(bundle, review, paper)
                    
                logger.info("Migrated extraction for %s", self.paper_id)
            except Exception as exc:
                logger.error("Failed to migrate extraction for %s: %s", self.paper_id, exc)
        
        # Migrate plan
        if plan_path and plan_path.exists():
            try:
                data = json.loads(plan_path.read_text(encoding="utf-8"))
                paper = PaperMetadata.model_validate(data["paper"])
                plan_review = data.get("plan_review", {"status": "approved", "notes": "Migrated from legacy"})
                
                from src.persistence import load_planner_envelope

                plan = load_planner_envelope(data)
                self.save_plan(plan, ReviewRecord.model_validate(plan_review), paper, str(extraction_path) if extraction_path else None)
                logger.info("Migrated plan for %s", self.paper_id)
            except Exception as exc:
                logger.error("Failed to migrate plan for %s: %s", self.paper_id, exc)
        
        # Migrate report
        if report_path and report_path.exists():
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
                paper = PaperMetadata.model_validate(data["paper"])
                report = ReviewerReport.model_validate(data["reviewer_report"])
                self.save_report(report, paper)
                logger.info("Migrated report for %s", self.paper_id)
            except Exception as exc:
                logger.error("Failed to migrate report for %s: %s", self.paper_id, exc)