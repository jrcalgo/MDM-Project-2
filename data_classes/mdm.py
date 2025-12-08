# data_classes/mdm.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal


# -------------------------------------------------------------------
# Input record coming from CSV/XLSX
# -------------------------------------------------------------------

@dataclass
class MDMData:
    name: str
    address: str
    city: str
    state: str
    country: str
    postal_code: str


# -------------------------------------------------------------------
# Derived queries used for tools / search
# -------------------------------------------------------------------

@dataclass
class QueryData:
    q_name: str
    q_name_geo: str
    q_full_addr: str


# -------------------------------------------------------------------
# Canonical enriched record from tools + LLM
# (not strictly required at runtime yet, but useful as a schema)
# -------------------------------------------------------------------

@dataclass
class OutputMDM:
    canonical_name: str
    aka: List[str]
    address: str
    city: str
    state: str
    postal_code: str
    country: str
    lat: Optional[float]
    lon: Optional[float]
    websites: List[str]
    ids: Dict[str, Any]
    confidence: float


# -------------------------------------------------------------------
# Full per-row result written to rawSearchResults.json
# -------------------------------------------------------------------

@dataclass
class OutputRow:
    row_index: int
    input: Dict[str, Any]          # original MDMData as dict
    queries: Dict[str, Any]        # QueryData as dict
    timing_ms: int
    success: bool
    message: str
    public_presence: bool
    websites: List[str]
    evidence_count: int
    notes: str
    enriched_mdm: Dict[str, Any]   # OutputMDM as dict

# ============ Dataclasses for LLM Output ============

@dataclass
class IssueDetail:
    """Detailed issue found during validation."""
    severity: str  # "critical", "high", "medium", "low"
    issue_type: str  # "missing_field", "hallucination", "poor_provenance", etc.
    field: str
    description: str
    evidence: str


@dataclass
class ConfidenceAssessment:
    """Assessment of the confidence score."""
    current_score: float
    is_appropriate: bool
    suggested_score: float
    reasoning: str


@dataclass
class ValidationScrutiny:
    """Complete LLM scrutiny response."""
    overall_quality: str  # "high", "medium", "low"
    provenance_quality_score: float  # 0.0-1.0
    issues: List[Dict[str, Any]]
    confidence_assessment: Dict[str, Any]
    recommendations: List[str]
    hallucination_detected: bool
    missing_fields_available: List[str]
    search_adequacy: str  # "sufficient", "insufficient", "poor"


# ============ Output Dataclass ============

@dataclass
class ValidationResult:
    """Output from validation process."""
    status: Literal["pass", "needs_refinement", "fail", "validation_done"]
    confidence_meets_threshold: bool
    threshold_used: float
    actual_confidence: float
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]
    validation_notes: str
    approved_for_csv: bool
    feedback_for_search: Optional[Dict[str, Any]]
    provenance_quality_score: float
    iteration_count: int