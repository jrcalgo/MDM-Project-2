# data_classes/mdm.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
