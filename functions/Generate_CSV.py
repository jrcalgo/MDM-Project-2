# functions/Generate_CSV.py

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional


class GenerateCSVTool:
    """
    Take:
      - input_rows: list of original input records (MDMData dataclasses or dicts)
      - processed_rows: list of raw OutputRow dicts
      - dnb_rows (optional): parallel list of DNB/PRIMARY dicts when --use-dnb is enabled

    Produce:
      - a single CSV with:
          * all input fields
          * enriched_mdm flattened as enriched_*
          * optional dnb_* columns
          * meta columns: row_index, success, message, evidence_count
    """

    def run(
        self,
        input_rows: List[Any],
        processed_rows: List[Dict[str, Any]],
        out_path: str,
        dnb_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        # ------------------------------
        # 1) Normalize input_rows -> dicts
        # ------------------------------
        norm_inputs: List[Dict[str, Any]] = []
        for r in input_rows:
            if is_dataclass(r):
                norm_inputs.append(asdict(r))
            elif isinstance(r, dict):
                norm_inputs.append(r)
            else:
                norm_inputs.append({"_raw_input": str(r)})

        # Make sure lengths line up reasonably
        n = min(len(norm_inputs), len(processed_rows))
        norm_inputs = norm_inputs[:n]
        processed_rows = processed_rows[:n]

        if dnb_rows is None:
            dnb_rows = [None] * n
        else:
            dnb_rows = (dnb_rows + [None] * n)[:n]

        # ------------------------------
        # 2) Discover all keys for columns
        # ------------------------------
        input_keys = set()
        enriched_keys = set()
        dnb_keys = set()

        for base, proc, dnb in zip(norm_inputs, processed_rows, dnb_rows):
            base = base or {}
            proc_d = proc if isinstance(proc, dict) else {}

            # Input fields (from the original MDM input)
            input_keys.update(base.keys())

            # Enriched MDM fields (from raw OutputRow)
            enriched = (
                proc_d.get("enriched_mdm")
                or proc_d.get("enrichedMDM")
                or proc_d.get("output_mdm")
                or {}
            )
            if isinstance(enriched, dict):
                enriched_keys.update(enriched.keys())

            # DNB fields (when present)
            if isinstance(dnb, dict):
                dnb_keys.update(dnb.keys())

        # Deterministic ordering
        input_cols = sorted(input_keys)
        enriched_cols = sorted(enriched_keys)
        dnb_cols = sorted(dnb_keys)

        # Meta columns we always include
        meta_cols = ["row_index", "success", "message", "evidence_count"]

        # ------------------------------
        # 3) Build the CSV header
        # ------------------------------
        header: List[str] = []
        header.extend(input_cols)
        header.extend(f"enriched_{k}" for k in enriched_cols)

        # Only add DNB columns if we actually have any DNB data
        has_any_dnb = any(isinstance(d, dict) and d for d in dnb_rows)
        if has_any_dnb:
            header.extend(f"dnb_{k}" for k in dnb_cols)

        header.extend(meta_cols)

        # ------------------------------
        # 4) Write rows
        # ------------------------------
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            for idx, (base, proc, dnb) in enumerate(zip(norm_inputs, processed_rows, dnb_rows)):
                row: Dict[str, Any] = {}

                base = base or {}
                proc_d = proc if isinstance(proc, dict) else {}

                enriched = (
                    proc_d.get("enriched_mdm")
                    or proc_d.get("enrichedMDM")
                    or proc_d.get("output_mdm")
                    or {}
                )
                if not isinstance(enriched, dict):
                    enriched = {}

                # ---- Input columns (from original MDMData)
                for k in input_cols:
                    row[k] = base.get(k, "")

                # ---- Enriched MDM columns
                for k in enriched_cols:
                    row[f"enriched_{k}"] = enriched.get(k, "")

                # ---- DNB columns (if any)
                if has_any_dnb and isinstance(dnb, dict):
                    for k in dnb_cols:
                        row[f"dnb_{k}"] = dnb.get(k, "")
                elif has_any_dnb:
                    # No DNB for this row => leave blanks
                    for k in dnb_cols:
                        row[f"dnb_{k}"] = ""

                # ---- Meta columns
                row["row_index"] = (
                    proc_d.get("row_index")
                    or proc_d.get("rowIndex")
                    or idx
                )
                row["success"] = proc_d.get("success", "")
                row["message"] = proc_d.get("message", "")
                row["evidence_count"] = (
                    proc_d.get("evidence_count")
                    or proc_d.get("evidenceCount")
                    or ""
                )

                writer.writerow(row)
