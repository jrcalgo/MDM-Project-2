# utils/report_builder.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json

# Column order for CSV + HTML table
REPORT_COLUMNS: List[str] = [
    "ROW_INDEX",
    "INPUT_NAME",
    "INPUT_ADDRESS",
    "INPUT_CITY",
    "INPUT_STATE",
    "INPUT_POSTAL_CODE",
    "INPUT_COUNTRY",
    "SEARCH_CANONICAL_NAME",
    "SEARCH_AKA",
    "SEARCH_ADDRESS",
    "SEARCH_CITY",
    "SEARCH_STATE",
    "SEARCH_POSTAL_CODE",
    "SEARCH_COUNTRY",
    "SEARCH_WEBSITES",
    "SEARCH_IDS",
    "SEARCH_SUCCESS",
    "SEARCH_MESSAGE",
    "SEARCH_EVIDENCE_COUNT",
    "SEARCH_CONFIDENCE",
    "VALIDATION_STATUS",
    "CONFIDENCE_MEETS_THRESHOLD",
    "THRESHOLD_USED",
    "ACTUAL_CONFIDENCE",
    "PROVENANCE_QUALITY_SCORE",
    # NOTE: APPROVED_FOR_CSV REMOVED FROM REPORT OUTPUT
    "VALIDATION_NOTES",
    "SOURCES",
]


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_merged_report(output_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Build merged report rows from:
      - input_MDM.json
      - rawSearchResults.json
      - validation_results.json

    Returns: list of dicts with keys from REPORT_COLUMNS.
    """
    output_dir = Path(output_dir)

    input_path = output_dir / "input_MDM.json"
    raw_path = output_dir / "rawSearchResults.json"
    validation_path = output_dir / "validation_results.json"

    input_mdm = _load_json(input_path)          # list of dicts
    raw_results = _load_json(raw_path)         # list of dicts
    # validation_results.json might not exist if validation never ran for some reason
    try:
        validation_data = _load_json(validation_path)
        validation_map = validation_data.get("results", {}) or {}
    except FileNotFoundError:
        validation_data = {}
        validation_map = {}

    rows: List[Dict[str, Any]] = []

    n = min(len(input_mdm), len(raw_results))

    for idx in range(n):
        input_row = input_mdm[idx] or {}
        raw_row = raw_results[idx] or {}
        enriched = raw_row.get("enriched_mdm") or {}

        # ---------------- Identity & Input data ----------------
        input_name = input_row.get("name", "") or ""
        record_key = input_name or f"record_{idx}"

        row: Dict[str, Any] = {
            "ROW_INDEX": idx,
            "INPUT_NAME": input_name,
            "INPUT_ADDRESS": input_row.get("address", "") or "",
            "INPUT_CITY": input_row.get("city", "") or "",
            "INPUT_STATE": input_row.get("state", "") or "",
            "INPUT_POSTAL_CODE": input_row.get("postal_code", "") or "",
            "INPUT_COUNTRY": input_row.get("country", "") or "",
        }

        # ---------------- Search / enriched fields ----------------
        aka = enriched.get("aka") or []
        if not isinstance(aka, list):
            aka = [aka]

        websites = enriched.get("websites") or raw_row.get("websites") or []
        if not isinstance(websites, list):
            websites = [websites]

        ids = enriched.get("ids") or {}
        if isinstance(ids, dict):
            ids_str = "; ".join(f"{k}: {v}" for k, v in ids.items())
        elif ids:
            ids_str = str(ids)
        else:
            ids_str = ""

        row.update(
            {
                "SEARCH_CANONICAL_NAME": enriched.get("canonical_name", "") or "",
                "SEARCH_AKA": "; ".join(str(a) for a in aka) if aka else "",
                "SEARCH_ADDRESS": enriched.get("address", "") or "",
                "SEARCH_CITY": enriched.get("city", "") or "",
                "SEARCH_STATE": enriched.get("state", "") or "",
                "SEARCH_POSTAL_CODE": enriched.get("postal_code", "") or "",
                "SEARCH_COUNTRY": enriched.get("country", "") or "",
                "SEARCH_WEBSITES": "; ".join(str(w) for w in websites) if websites else "",
                "SEARCH_IDS": ids_str,
                "SEARCH_SUCCESS": bool(raw_row.get("success", False)),
                "SEARCH_MESSAGE": raw_row.get("message", "") or "",
                "SEARCH_EVIDENCE_COUNT": int(raw_row.get("evidence_count") or 0),
                "SEARCH_CONFIDENCE": float(
                    enriched.get("confidence", 0.0) or 0.0
                ),
            }
        )

        # ---------------- Validation info ----------------
        v = validation_map.get(record_key) or validation_map.get(f"record_{idx}") or {}

        status = v.get("status", "unknown")
        meets_threshold = bool(v.get("confidence_meets_threshold", False))
        threshold_used = float(v.get("threshold_used", 0.0) or 0.0)
        actual_conf = float(v.get("actual_confidence", row["SEARCH_CONFIDENCE"]) or 0.0)
        prov_score = float(v.get("provenance_quality_score", 0.0) or 0.0)
        # approved_for_csv is still in validation results, but we don't surface it
        # in the merged report columns anymore.
        notes = v.get("validation_notes", "") or ""

        row.update(
            {
                "VALIDATION_STATUS": status,
                "CONFIDENCE_MEETS_THRESHOLD": meets_threshold,
                "THRESHOLD_USED": threshold_used,
                "ACTUAL_CONFIDENCE": actual_conf,
                "PROVENANCE_QUALITY_SCORE": prov_score,
                "VALIDATION_NOTES": notes,
            }
        )

        # ---------------- Sources (from rawResults only) ----------------
        # Show sources for "pass", "needs_refinement", and "validation_done".
        # Hide for "fail" or "unknown".
        if status in ("pass", "needs_refinement", "validation_done"):
            # Try multiple locations for websites / provenance
            src_websites = (
                raw_row.get("websites")
                or enriched.get("websites")
                or raw_row.get("search_metadata", {}).get("websites")
                or []
            )
            if not isinstance(src_websites, list):
                src_websites = [src_websites]
            sources_str = "; ".join(str(w) for w in src_websites) if src_websites else ""
        else:
            sources_str = ""

        row["SOURCES"] = sources_str

        # Ensure all columns exist (fill missing with empty/False/0)
        for col in REPORT_COLUMNS:
            if col not in row:
                row[col] = ""  # default

        rows.append(row)

    return rows
