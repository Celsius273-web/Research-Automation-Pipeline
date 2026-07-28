"""State models for the phased LangGraph pipeline."""

from __future__ import annotations

from typing import Any, Literal, Generic, TypeVar
from typing_extensions import TypedDict

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


SECTION_NAMES = [
    "abstract",
    "method",
    "experiments",
    "hyperparameters",
    "appendix",
]


# Unified agent output envelope schema
class UnknownItem(BaseModel):
    """Represents a missing or uncertain field in agent output."""
    field: str
    reason: str
    severity: Literal["low", "medium", "high"] = "medium"


PayloadT = TypeVar("PayloadT")


class AgentEnvelope(BaseModel, Generic[PayloadT]):
    """Shared top-level contract for all LLM-backed agents."""
    schema_version: str = "2.0"
    agent: Literal["analyst", "planner", "engineer", "reviewer"]
    status: Literal["ok", "partial", "blocked", "ready_to_execute"] = "ok"
    unknowns: list[UnknownItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    payload: PayloadT


class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    pdf_path: str
    arxiv_id: str | None = None


class IngestedPaperRecord(BaseModel):
    """Model for papers registered via ingestion pipeline."""
    paper_id: str
    title: str
    arxiv_id: str | None = None
    repo_url: str | None = None
    bundle_path: str
    pdf_path: str
    code_path: str | None = None
    pdf_checksum: str
    ingested_at: datetime
    repo_commit_sha: str | None = None


class SectionTextMap(BaseModel):
    abstract: str = ""
    method: str = ""
    experiments: str = ""
    hyperparameters: str = ""
    appendix: str = ""
    full_text: str = ""


# Analyst payload schemas
class AnalystPayloadCore(BaseModel):
    """Required fields for analyst extraction."""
    research_question: str
    methodology: str
    datasets: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    evaluation_metrics: list[str] = Field(default_factory=list)


class AnalystPayloadExtensions(BaseModel):
    """Optional enrichment fields for analyst extraction."""
    hyperparameters: dict[str, str] = Field(default_factory=dict)
    reported_results: list["ReportedResult"] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    notes: str = ""


class AnalystPayload(BaseModel):
    """Complete analyst payload with core and extensions."""
    core: AnalystPayloadCore
    extensions: AnalystPayloadExtensions = Field(default_factory=AnalystPayloadExtensions)


class SectionExtraction(BaseModel):
    """Flat per-section extraction produced by the Paper Analyst."""

    model_config = ConfigDict(extra="forbid")

    research_question: str = ""
    methodology: str = ""
    datasets_or_benchmarks: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, str] = Field(default_factory=dict)
    evaluation_metrics: list[str] = Field(default_factory=list)
    reported_results: list["ReportedResult"] = Field(default_factory=list)
    notes: str = ""


class ExtractionBundle(BaseModel):
    by_section: dict[str, SectionExtraction] = Field(default_factory=dict)
    merged: SectionExtraction = Field(default_factory=SectionExtraction)


class ReviewRecord(BaseModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    notes: str = ""


class PlannerAnalystOutput(BaseModel):
    """Analyst fields accepted by the Planner's unified input contract."""

    model_config = ConfigDict(extra="forbid")

    research_question: str
    methodology: str
    datasets_or_benchmarks: list[str]
    variables: list[str]
    hyperparameters: dict[str, Any]
    evaluation_metrics: list[str]
    reported_results: list["ReportedResult"]
    notes: str


class PlannerRepoContext(BaseModel):
    """Repository facts exposed to the Planner LLM."""

    model_config = ConfigDict(extra="forbid")

    url: str
    language: str
    build_system: str
    has_code: bool
    setup_time_minutes: float
    file_tree: str
    readme_summary: str
    example_commands: list[str] = Field(default_factory=list)


class PlannerFlags(BaseModel):
    """Derived routing facts for the Planner."""

    model_config = ConfigDict(extra="forbid")

    has_research_question: bool
    has_methodology: bool
    has_code_repo: bool
    has_datasets: bool
    paper_type: Literal["methods", "empirical", "toolkit"]


class UnifiedPlannerInput(BaseModel):
    """Fixed four-key Planner input contract."""

    model_config = ConfigDict(extra="forbid")

    analyst_output: PlannerAnalystOutput
    repo_context: PlannerRepoContext
    paper_context: PaperMetadata
    flags: PlannerFlags


class PlannerInputContext(BaseModel):
    paper: PaperMetadata
    approved_extraction: SectionExtraction
    extraction_sections: dict[str, SectionExtraction] | None = None
    runtime_constraints: dict[str, str] = Field(default_factory=dict)
    repo_context: dict[str, Any] = Field(default_factory=dict)
    repo_setup_guide: str = ""
    hyperparameter_reference: str = ""
    extraction_file_path: str | None = None
    paper_bundle_path: str | None = None


class PlanStepCore(BaseModel):
    """Core required fields for a plan step."""
    step_id: str
    title: str
    goal: str
    run_command: str = ""
    depends_on: list[str] = Field(default_factory=list)
    results_path: str = ""


class PlanStepExtensions(BaseModel):
    """Optional enrichment fields for a plan step."""
    actions: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)


class PlanStepComplete(BaseModel):
    """Complete plan step with core and extensions."""
    core: PlanStepCore
    extensions: PlanStepExtensions = Field(default_factory=PlanStepExtensions)


class PlannerPayloadCore(BaseModel):
    """Required fields for planner output."""
    plan_summary: str
    domain: str
    objective: str
    steps: list[PlanStepCore] = Field(default_factory=list)


class PlannerPayloadExtensions(BaseModel):
    """Optional enrichment fields for planner output."""
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
    experiment_matrix: list["ExperimentSpec"] = Field(default_factory=list)
    verification_checks: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class PlannerPayload(BaseModel):
    """Complete planner payload with core and extensions."""
    core: PlannerPayloadCore
    extensions: PlannerPayloadExtensions = Field(default_factory=PlannerPayloadExtensions)


class PlanStep(BaseModel):
    """Legacy model for backward compatibility during migration."""
    step_id: str
    title: str
    goal: str = ""
    actions: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    results_path: str = ""
    depends_on: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)


class ExperimentSpec(BaseModel):
    name: str
    target: str = ""
    variables: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=list)
    source_section: str = ""
    implementation_steps: list[str] = Field(default_factory=list)
    execution_pattern: str = ""
    expected_runtime_minutes: int | None = None


class ExecutionPlan(BaseModel):
    schema_version: str = "1.0"
    plan_summary: str = ""
    domain: str = ""
    objective: str = ""
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    experiment_matrix: list[ExperimentSpec] = Field(default_factory=list)
    verification_checks: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class PlanReviewRecord(BaseModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    notes: str = ""
    edited_fields: list[str] = Field(default_factory=list)


class RepoContext(BaseModel):
    repo_url: str = ""
    repo_path: str = ""
    language: str = "unknown"
    build_system: str = "unknown"
    has_code: bool = False
    setup_time_minutes: float = 0
    file_tree: str = ""
    readme_summary: str = ""
    example_commands: list[str] = Field(default_factory=list)
    notes: str = ""


class PatchProposal(BaseModel):
    file_path: str
    action: Literal["create", "modify", "delete"]
    content: str = ""
    rationale: str = ""


class FailureContext(BaseModel):
    stage: Literal["build", "runtime", "timeout"] = "runtime"
    exit_code: int | None = None
    log_excerpt: str = ""
    prior_patch_summary: str = ""


class EngineerInputContext(BaseModel):
    paper: PaperMetadata
    execution_plan: ExecutionPlan | AgentEnvelope[PlannerPayload]
    plan_step: PlanStep | PlanStepCore
    repo_context: RepoContext
    runtime_constraints: dict[str, str] = Field(default_factory=dict)
    failure_context: FailureContext | None = None


class EngineerPayloadCore(BaseModel):
    """Required fields for engineer output."""
    step_id: str
    detected_language: str
    patches: list["PatchProposal"] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)


class EngineerPayloadExtensions(BaseModel):
    """Optional enrichment fields for engineer output."""
    rationale: str = ""
    missing_context: list[str] = Field(default_factory=list)
    risk_analysis: list[str] = Field(default_factory=list)


class EngineerPayload(BaseModel):
    """Complete engineer payload with core and extensions."""
    core: EngineerPayloadCore
    extensions: EngineerPayloadExtensions = Field(default_factory=EngineerPayloadExtensions)


class EngineerOutput(BaseModel):
    """Legacy model for backward compatibility during migration."""
    step_id: str = ""
    patches: list["PatchProposal"] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    rationale: str = ""
    missing_context: list[str] = Field(default_factory=list)


class EngineerReviewRecord(BaseModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    notes: str = ""
    edited_fields: list[str] = Field(default_factory=list)


class MetricResult(BaseModel):
    benchmark: str = ""
    metric_name: str
    value: str
    source_path: str = ""


class RunAttempt(BaseModel):
    attempt_number: int = 0
    step_id: str = ""
    stage: Literal["build", "runtime", "timeout"] = "runtime"
    command: str = ""
    exit_code: int = 1
    success: bool = False
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    logs_path: str = ""
    failure_type: Literal["build_error", "runtime_error", "timeout", "none"] = "runtime_error"


class ExecutorResult(BaseModel):
    attempts: list[RunAttempt] = Field(default_factory=list)
    captured_metrics: list[MetricResult] = Field(default_factory=list)
    final_status: Literal["success", "failed", "exhausted_retries", "pending"] = "pending"
    total_attempts: int = 0
    needs_manual_patch: bool = False


class ReportedResult(BaseModel):
    benchmark: str = ""
    metric_name: str
    value: str
    source: str = ""


class ComparisonRow(BaseModel):
    metric_name: str
    benchmark: str = ""
    reported_value: str = ""
    reproduced_value: str = ""
    absolute_difference: float | None = None
    relative_difference_pct: float | None = None
    match_status: Literal[
        "match",
        "close",
        "diverged",
        "missing_reproduced",
        "missing_reported",
        "unparsable",
    ]


class ReviewerPayloadCore(BaseModel):
    """Required fields for reviewer output."""
    summary: str
    verdict: Literal["reproduced", "partially_reproduced", "not_reproduced", "inconclusive"]
    comparison_rows: list["ComparisonRow"] = Field(default_factory=list)


class ReviewerPayloadExtensions(BaseModel):
    """Optional enrichment fields for reviewer output."""
    reproduction_rate: float = 0.0
    risks: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    run_summary: dict[str, str] = Field(default_factory=dict)
    deep_diagnostics: list[str] = Field(default_factory=list)


class ReviewerPayload(BaseModel):
    """Complete reviewer payload with core and extensions."""
    core: ReviewerPayloadCore
    extensions: ReviewerPayloadExtensions = Field(default_factory=ReviewerPayloadExtensions)


class ReviewerReport(BaseModel):
    """Legacy model for backward compatibility during migration."""
    schema_version: str = "1.0"
    paper_id: str = ""
    domain: str = ""
    summary: str = ""
    verdict: Literal["reproduced", "partially_reproduced", "not_reproduced", "inconclusive"] = (
        "inconclusive"
    )
    reproduction_rate: float = 0.0
    comparison_table: list[ComparisonRow] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    run_summary: dict[str, str] = Field(default_factory=dict)


class ResearchState(TypedDict, total=False):
    paper: PaperMetadata
    section_texts: SectionTextMap
    extraction: ExtractionBundle
    review: ReviewRecord
    approved_extraction: SectionExtraction
    planner_output: ExecutionPlan | AgentEnvelope[PlannerPayload]
    planner_output_json: dict[str, Any]
    plan_review: PlanReviewRecord
    repo_context: RepoContext
    engineer_output: EngineerOutput
    engineer_review: EngineerReviewRecord
    executor_result: ExecutorResult
    retry_count: int
    reviewer_report: ReviewerReport
    errors: list[str]


def make_initial_state(paper: PaperMetadata) -> ResearchState:
    """Build a complete baseline state with placeholders for future phases."""
    return {
        "paper": paper,
        "section_texts": SectionTextMap(),
        "extraction": ExtractionBundle(),
        "review": ReviewRecord(status="pending"),
        "approved_extraction": SectionExtraction(),
        "planner_output": ExecutionPlan(),
        "planner_output_json": {},
        "plan_review": PlanReviewRecord(status="pending"),
        "repo_context": RepoContext(),
        "engineer_output": EngineerOutput(),
        "engineer_review": EngineerReviewRecord(status="pending"),
        "executor_result": ExecutorResult(),
        "retry_count": 0,
        "reviewer_report": ReviewerReport(),
        "errors": [],
    }
