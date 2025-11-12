from typing import Any, Dict
from difflib import SequenceMatcher
import numbers
import re
import json


class ComparisonFunction():
    """
    ComparisonFunction.py

    Compares company customer records between original and updated data,
    identifying discrepancies and calculating similarity scores.

    Usage:
        report = compare_records(original_record, search_record)
    """

    def __init__(self):
        pass

    def _normalize_str(self, s: str):
        if not isinstance(s, str):
            return str(s)
        s = s.strip()

        s = re.sub(r"\s+", " ", s)
        s = s.lower()
        return s

    def _similarity(self, a, b):
        """
        Returns a float in [0,1] representing similarity between a and b.
        - For numbers: 1 - normalized absolute difference (clamped)
        - For strings: difflib.SequenceMatcher ratio on normalized strings
        - For other types: equality -> 1.0 else 0.0
        """
        if a is None and b is None:
            return 1.0
        if a is None or b is None:
            return 0.0

        if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            try:
                if a == b:
                    return 1.0
                diff = abs(float(a) - float(b))
                denom = max(abs(float(a)), abs(float(b)), 1.0)
                score = 1.0 - (diff / denom)
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.0

        if isinstance(a, str) and isinstance(b, str):
            na = self._normalize_str(a)
            nb = self._normalize_str(b)
            if na == nb:
                return 1.0
            # SequenceMatcher is fine for simple fuzzy scoring
            return SequenceMatcher(None, na, nb).ratio()

        # Fallback: convert to str and compare
        try:
            sa = self._normalize_str(str(a))
            sb = self._normalize_str(str(b))
            if sa == sb:
                return 1.0
            return SequenceMatcher(None, sa, sb).ratio()
        except ValueError:
            return 1.0 if a == b else 0.0

    def _issue_for_values(self, orig, found, sim: float):
        """
        Heuristic reasons for mismatch. Returns a short label describing likely issue.
        """
        if orig is None and found is not None:
            return "unexpected_value_present"
        if orig is not None and found is None:
            return "missing_value"
        if isinstance(orig, numbers.Number) and isinstance(found, numbers.Number):
            if sim == 1.0:
                return None
            return "numeric_difference"
        if isinstance(orig, str) and isinstance(found, str):
            no = self._normalize_str(orig)
            nf = self._normalize_str(found)
            if no == nf and orig != found:
                return "format_case_whitespace_difference"
            if sim > 0.85:
                return "minor_string_variation"
            return "string_mismatch"
        if sim > 0.8:
            return "minor_variation"
        return "value_mismatch"

    def _compare_scalar(self, orig, found):
        sim = self._similarity(orig, found)
        issue = None if sim == 1.0 else self._issue_for_values(
            orig, found, sim)
        return {
            "original": orig,
            "found": found,
            "match": sim == 1.0,
            "similarity": round(sim, 4),
            "issue": issue,
        }

    def _best_match_in_list(self, item, candidates):
        best = None
        best_score = -1.0
        for c in candidates:
            s = self._similarity(item, c)
            if s > best_score:
                best_score = s
                best = c
        return best, best_score

    def _compare_list(self, orig_list, found_list):
        # Try one-to-one by index first
        items = []
        scores = []
        if not isinstance(found_list, list):
            return {
                "original": orig_list,
                "found": found_list,
                "match": False,
                "similarity": 0.0,
                "issue": "type_mismatch_expected_list",
                "details": [],
            }
        # If lengths equal, compare by index
        if len(orig_list) == len(found_list):
            for i, (o, f) in enumerate(zip(orig_list, found_list)):
                items.append(
                    {
                        "index": i,
                        "result": self._compare_any(o, f),
                    }
                )
                scores.append(items[-1]["result"]["similarity"])
        else:
            # For each original item find best candidate
            remaining = found_list[:]
            for i, o in enumerate(orig_list):
                best, _ = self._best_match_in_list(
                    o, remaining) if remaining else (None, 0.0)  # best_score
                items.append(
                    {
                        "index": i,
                        "result": self._compare_any(o, best),
                        "matched_to": best,
                    }
                )
                scores.append(items[-1]["result"]["similarity"])
                if best in remaining:
                    remaining.remove(best)
            # any leftover found_list items are extras
            extras = [x for x in found_list if x not in [
                it.get("matched_to") for it in items]]
            if extras:
                items.append({"extra_found_items": extras})

        avg = float(sum(scores)) / len(scores) if scores else 0.0
        return {
            "original": orig_list,
            "found": found_list,
            "match": avg == 1.0,
            "similarity": round(avg, 4),
            "issue": None if avg == 1.0 else "list_mismatch",
            "details": items,
        }

    def _compare_dict(self, orig, found):
        details = {}
        sims = []
        if not isinstance(found, dict):
            # Type mismatch: original is dict but found isn't
            return {
                "original": orig,
                "found": found,
                "match": False,
                "similarity": 0.0,
                "issue": "type_mismatch_expected_dict",
                "details": {},
            }
        for k, v in orig.items():
            found_val = found.get(k, None)
            res = self._compare_any(v, found_val)
            details[k] = res
            sims.append(res["similarity"])
        # Also record extra keys in found
        extra_keys = [k for k in found.keys() if k not in orig.keys()]
        if extra_keys:
            details["_extra_found_keys"] = {k: found[k] for k in extra_keys}
        avg = float(sum(sims)) / len(sims) if sims else 0.0
        return {
            "original": orig,
            "found": found,
            "match": avg == 1.0,
            "similarity": round(avg, 4),
            "issue": None if avg == 1.0 else "record_mismatch",
            "details": details,
        }

    def _compare_any(self, orig, found):
        # Detect simple container types first
        if isinstance(orig, dict):
            return self._compare_dict(orig, found)
        if isinstance(orig, list):
            return self._compare_list(orig, found)
        # otherwise scalar
        return self._compare_scalar(orig, found)

    def compare_records(self, original_record, search_record):
        """
        Compare original_record against search_record and produce a report.

        Returns a dictionary:
        {
        "summary": {
            "overall_similarity": float,
            "match": bool,
            "fields_compared": int
        },
        "comparison": { ... per-field results ... }
        }

        The function is tolerant of nested structures. For dicts it compares keys present
        in original_record to values in search_record by the same key. Lists attempt
        indexwise or best-match comparison. Scalars are fuzzy-compared using difflib
        for strings and normalized difference for numbers.
        """
        comparison = self._compare_any(original_record, search_record)
        # Pull out an overall similarity/summary
        overall_similarity = comparison.get("similarity", 0.0)
        # count fields compared heuristically

        def _count_fields(node: Dict[str, Any]) -> int:
            if not isinstance(node, dict):
                return 0
            if "details" in node and isinstance(node["details"], dict):
                # dict case
                cnt = 0
                for k, v in node["details"].items():
                    if k.startswith("_"):
                        continue
                    cnt += _count_fields(v) if isinstance(v, dict) else 1
                return cnt
            if "details" in node and isinstance(node["details"], list):
                return len([d for d in node["details"] if isinstance(d, dict)])
            # scalar
            if "original" in node and "found" in node:
                return 1
            return 0

        fields_compared = _count_fields(comparison)

        report = {
            "summary": {
                "overall_similarity": round(overall_similarity, 4),
                "match": comparison.get("match", False),
                "fields_compared": fields_compared,
            },
            "comparison": comparison,
        }

        return report

if __name__ == "__main__":
    CF = ComparisonFunction()
    orig_data = {
        "name": "John A. Smith",
        "email": "john.smith@example.com",
        "age": 42,
        "addresses": [
            {"line1": "123 Main St", "city": "Charlotte"},
            {"line1": "PO Box 456", "city": "Charlotte"},
        ],
    }
    found_data = {
        "name": "john smith",
        "email": "john.smith@example.com ",
        "age": 41,
        "addresses": [
            {"line1": "123 Main Street", "city": "Charlotte"},
        ],
        "confidence": 0.78,
    }
    print(json.dumps(CF.compare_records(orig_data, found_data), indent=2))
