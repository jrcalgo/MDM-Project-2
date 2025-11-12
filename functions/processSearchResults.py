# processSearchResults.py
# Library-only: transform the results.json produced by driver.py into:
# { "searchResults": [ { ... }, { "search": false }, ... ] }

from __future__ import annotations
import json
from typing import Any, Dict, List

def _string_or_empty(v: Any) -> str:
    return v if isinstance(v, str) else ""

def _is_effectively_empty_enriched(enriched: Any) -> bool:
    """Treat enriched_mdm as empty unless at least one key field is populated."""
    if not isinstance(enriched, dict) or not enriched:
        return True
    key_fields = ["canonical_name", "address", "city", "state", "postal_code", "country"]
    for k in key_fields:
        val = enriched.get(k)
        if isinstance(val, str) and val.strip():
            return False
    # also count as not-empty if websites/ids exist and non-empty
    if isinstance(enriched.get("websites"), list) and enriched["websites"]:
        return False
    if isinstance(enriched.get("ids"), dict) and enriched["ids"]:
        return False
    return True

def _format_success_obj(enriched: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map enriched_mdm -> required output fields.
    Include ALL fields even if empty (initialize to "").
    canonical_name -> name
    """
    return {
        "name": _string_or_empty(enriched.get("canonical_name")),
        "address": _string_or_empty(enriched.get("address")),
        "city": _string_or_empty(enriched.get("city")),
        "state": _string_or_empty(enriched.get("state")),
        "country": _string_or_empty(enriched.get("country")),
        "postal_code": _string_or_empty(enriched.get("postal_code")),
        "search": True,
    }

def process_results(input_json_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read the list produced by driver.py and return:
      { "searchResults": [ per-record objects ] }
    - If success==True AND enriched_mdm has meaningful info -> include fields + search:true
    - Else -> { "search": false }
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[Dict[str, Any]] = []
    for item in (data or []):
        success = bool(item.get("success"))
        enriched = item.get("enriched_mdm") or {}

        if success and not _is_effectively_empty_enriched(enriched):
            out.append(_format_success_obj(enriched))
        else:
            out.append({"search": False})

    return {"searchResults": out}
