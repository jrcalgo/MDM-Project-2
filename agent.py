import argparse
import csv
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, TypedDict, Annotated

from data_classes.mdm import MDMData
from functions.searchFunction import run_batch
from utils.read_input_records import read_input_rows
from functions.processSearchResults import process_results
from functions.ComparisonFunction import ComparisonFunction
from functions.ValidatorFunction import ValidatorFunction, ValidationResult
import functions.ValidatorFunction as validator_module
from functions.Generate_CSV import GenerateCSVTool

from langgraph.graph import StateGraph, START, END

comparisonTool = ComparisonFunction().compare_records
validationTool = ValidatorFunction().validate_record

from dotenv import load_dotenv
load_dotenv()



# ------------------------------------------------------------
# State definition
# ------------------------------------------------------------

class PipelineState(TypedDict):
    inputData: Annotated[List[MDMData], "list of input records as dataclasses"]          # list of input records as dataclasses
    rawResults: Annotated[List[Dict[str, Any]], "list of asdict(OutputRow)"]  # list of asdict(OutputRow)
    processedResults: Annotated[Dict[str, Any], "processedSearchResults.json structure"]  # processedSearchResults.json structure
    rawSearchResultsPath: Annotated[str, "path to raw search results"]
    processedSearchResultsPath: Annotated[str, "path to processed search results"]

    validationLoopCount: Annotated[int, "number of validation loops completed"]
    validatorDecision: Annotated[str, "validator route decision: sufficient, OR insufficient, OR failure"]
    validatorResults: Annotated[Dict[str, ValidationResult], "validation results for each analyzed record"]
    recordsToRefine: Annotated[List[int], "indices of records that need refinement"]
    previouslyFailedIndices: Annotated[List[int], "indices of records that failed in a previous iteration"]

    dnbData: Annotated[List[Dict[str, Any]], "parallel list of DNB dicts (can be empty)"]   # parallel list of DNB dicts (can be empty)
    useDnb: Annotated[bool, "whether we are using DNB in this run"]                    # whether we are using DNB in this run


# ------------------------------------------------------------
# Unified node (search + process)
# ------------------------------------------------------------

def search_and_process_node(state: PipelineState) -> PipelineState:
    records = state.get("inputData", [])
    use_dnb = state.get("useDnb", False)
    dnb_data = state.get("dnbData", [])
    raw_path = state.get("rawSearchResultsPath", "./data/rawSearchResults.json")
    processed_path = state.get("processedSearchResultsPath", "./data/processedSearchResults.json")
    
    records_to_refine = state.get("recordsToRefine", [])
    validation_results = state.get("validatorResults", {})
    existing_raw_results = state.get("rawResults", [])

    raw_path_obj = Path(raw_path)
    processed_path_obj = Path(processed_path)

    print("[INFO] Starting unified search + process node")
    
    # -----------------------------------------
    # Determine if this is initial search or refinement
    # -----------------------------------------
    is_refinement = len(records_to_refine) > 0 and len(existing_raw_results) > 0
    
    if is_refinement:
        print(f"[INFO] REFINEMENT MODE: Processing {len(records_to_refine)} record(s) needing refinement")
        
        # Build refinement feedback dict from validation results
        refinement_feedback: Dict[int, Dict[str, Any]] = {}
        for new_idx, original_idx in enumerate(records_to_refine):
            if original_idx < len(records):
                record_name = records[original_idx].name or f"record_{original_idx}"
                # Find validation result by record name
                val_result = None
                for key, vr in validation_results.items():
                    if key == record_name or key == f"record_{original_idx}":
                        val_result = vr
                        break
                
                if val_result and val_result.feedback_for_search:
                    feedback = val_result.feedback_for_search.copy()
                    feedback["previous_confidence"] = val_result.actual_confidence
                    feedback["target_threshold"] = val_result.threshold_used
                    refinement_feedback[new_idx] = feedback
                    print(f"[INFO]   Record {original_idx} ({record_name}): confidence={val_result.actual_confidence:.3f}, target={val_result.threshold_used:.1f}")
        
        # Filter records and DNB data to only those needing refinement
        records_to_process = [records[i] for i in records_to_refine if i < len(records)]
        dnb_to_process = [dnb_data[i] if i < len(dnb_data) else None for i in records_to_refine]
        
        # Run batch with refinement feedback
        try:
            raw_objs = run_batch(
                records_to_process,
                minimal_logging=True,
                use_dnb=use_dnb,
                dnb_rows=dnb_to_process,
                refinement_feedback=refinement_feedback,
            )
        except Exception as e:
            print(f"[ERROR] searchFunction.run_batch failed during refinement: {e}", file=sys.stderr)
            raise
        
        # Normalize to dicts
        refined_results = [
            asdict(r) if not isinstance(r, dict) else r
            for r in raw_objs
        ]
        
        # Merge refined results back into existing results at correct indices
        raw_results = existing_raw_results.copy()
        for refine_idx, original_idx in enumerate(records_to_refine):
            if original_idx < len(raw_results) and refine_idx < len(refined_results):
                raw_results[original_idx] = refined_results[refine_idx]
                print(f"[INFO] Updated result for record {original_idx}")
        
        # Write updated raw results
        try:
            with raw_path_obj.open("w", encoding="utf-8") as f:
                json.dump(raw_results, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Updated raw search results written to {raw_path_obj}")
        except Exception as e:
            print(f"[ERROR] Failed to write updated raw results: {e}", file=sys.stderr)
            raise
            
    else:
        # Initial search mode
        print(f"[INFO] INITIAL SEARCH MODE: Processing {len(records)} record(s)...\n")
        
        # -----------------------------------------
        # Step 1: Get raw results (search step, with skip logic)
        # -----------------------------------------
        if raw_path_obj.exists():
            print(f"[INFO] Found existing raw search results at {raw_path_obj}. Skipping search step.")
            try:
                with raw_path_obj.open("r", encoding="utf-8") as f:
                    raw_results = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to read existing raw results from {raw_path_obj}: {e}", file=sys.stderr)
                raise
        else:
            print("[INFO] No existing raw results found. Running searchFunction.run_batch...")
            try:
                raw_objs = run_batch(
                    records,
                    minimal_logging=True,
                    use_dnb=use_dnb,
                    dnb_rows=dnb_data,
                )
            except Exception as e:
                print(f"[ERROR] searchFunction.run_batch failed: {e}", file=sys.stderr)
                raise

            # Normalize to dicts for JSON + downstream use
            raw_results = [
                asdict(r) if not isinstance(r, dict) else r
                for r in raw_objs
            ]

            # -----------------------------------------
            # Step 2: Write raw results
            # -----------------------------------------
            try:
                with raw_path_obj.open("w", encoding="utf-8") as f:
                    json.dump(raw_results, f, indent=2, ensure_ascii=False)
                print(f"[INFO] Raw search results written to {raw_path_obj}")
            except Exception as e:
                print(f"[ERROR] Failed to write raw results: {e}", file=sys.stderr)
                raise

    # -----------------------------------------
    # Step 3: Get processed results (post-processing)
    # -----------------------------------------
    # Always regenerate processed results since raw results may have changed
    if is_refinement or not processed_path_obj.exists():
        print("[INFO] Running process_results...")
        try:
            processed = process_results(str(raw_path_obj))
        except Exception as e:
            print(f"[ERROR] process_results failed: {e}")
            raise

        # -----------------------------------------
        # Step 4: Write processed results
        # -----------------------------------------
        try:
            with processed_path_obj.open("w", encoding="utf-8") as f:
                json.dump(processed, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Processed results written to {processed_path_obj}")
        except Exception as e:
            print(f"[ERROR] Failed to write processed results: {e}")
            raise
    else:
        print(f"[INFO] Found existing processed results at {processed_path_obj}. Skipping process_results step.")
        try:
            with processed_path_obj.open("r", encoding="utf-8") as f:
                processed = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read existing processed results from {processed_path_obj}: {e}", file=sys.stderr)
            raise

    # -----------------------------------------
    # Step 5: Compare Input vs Processed Results
    # -----------------------------------------

    try:
        # Convert input MDMData objects to plain dicts for comparison / JSON
        input_dicts = [asdict(r) if not isinstance(r, dict) else r for r in records]

        # processed is expected to be a dict with key "searchResults"
        processed_list = processed.get("searchResults", []) if isinstance(processed, dict) else processed

        comparison = comparisonTool(input_dicts, processed_list)
        print("\n[INFO] Comparison of Input vs Processed Results:")
        # json.dumps(comparison, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] ComparisonFunction failed: {e}")
        raise

    # -----------------------------------------
    # Step 6: Generate comparison CSV (non-fatal if it fails)
    # -----------------------------------------

    try:
        csv_tool = GenerateCSVTool()

        # Reuse input_dicts from above
        input_dicts = [asdict(r) if not isinstance(r, dict) else r for r in records]

        # Use RAW results for the CSV because they contain enriched_mdm + meta flags
        raw_dicts = [
            r if isinstance(r, dict) else asdict(r)
            for r in raw_results
        ]

        # Build CSV path next to processed JSON output
        csv_path = processed_path_obj.with_name("comparison_input_output.csv")
        csv_tool.run(
            input_rows=input_dicts,
            processed_rows=raw_dicts,
            out_path=str(csv_path),
            dnb_rows=dnb_data if use_dnb else None,
        )
        print(f"[INFO] Comparison CSV written to {csv_path}")
    except Exception as e:
        print(f"[WARN] Failed to write comparison CSV: {e}", file=sys.stderr)


    # -----------------------------------------
    # Step 7: Return updated state
    # -----------------------------------------
    return {
        **state,
        "rawResults": raw_results,
        "processedResults": processed,
    }

def validator_node(state: PipelineState) -> PipelineState:
    """
    Validate search results using ValidatorFunction.

    Reads input_MDM.json and comparison_input_output.csv, then validates each
    record using the ValidatorFunction. Aggregates results to determine overall
    decision for routing.
    """
    raw_results = state.get("rawResults", [])
    loop_count = state.get("validationLoopCount", 0)
    processed_path = state.get("processedSearchResultsPath", "./data/processedSearchResults.json")

    processed_path_obj = Path(processed_path)
    output_dir = processed_path_obj.parent

    print(f"\n[INFO] Validator node starting (iteration {loop_count})...")
    print(f"[INFO] Validating {len(raw_results)} record(s)...")

    # Handle empty results
    if not raw_results:
        print("[WARN] No raw results to validate")
        return {
            **state,
            "validatorDecision": "fail",
            "validatorResults": {},
            "validationLoopCount": loop_count + 1,
        }

    # ---- Load input_MDM.json (not strictly required, but useful/logging) ----
    input_mdm_path = output_dir / "input_MDM.json"
    input_records = []
    if input_mdm_path.exists():
        try:
            with input_mdm_path.open("r", encoding="utf-8") as f:
                input_records = json.load(f)
            print(f"[INFO] Loaded {len(input_records)} input records from {input_mdm_path}")
        except Exception as e:
            print(f"[WARN] Failed to read input_MDM.json: {e}")
    else:
        print(f"[WARN] input_MDM.json not found at {input_mdm_path}")

    # ---- Load comparison_input_output.csv for LLM comparison context ----
    csv_path = output_dir / "comparison_input_output.csv"
    comparison_reports = _parse_comparison_csv(csv_path)
    print(f"[INFO] Loaded {len(comparison_reports)} comparison reports from CSV")

    validator_results: Dict[str, ValidationResult] = {}
    decisions: List[str] = []
    records_needing_refinement: List[int] = []

    # Get previously failed indices, these won't get another chance
    previously_failed = state.get("previouslyFailedIndices", [])
    new_failures: List[int] = []

    for idx, result in enumerate(raw_results):
        input_data = result.get("input", {})
        record_id = input_data.get("name", f"record_{idx}")

        comparison_report = None
        if idx < len(comparison_reports):
            comparison_report = comparison_reports[idx]

        print(f"[INFO] Validating: {record_id}")

        try:
            validation_result = validationTool(
                search_output=result,
                comparison_report=comparison_report,
                iteration=loop_count,
            )

            validator_results[record_id] = validation_result
            decisions.append(validation_result.status)

            if validation_result.status == "needs_refinement":
                records_needing_refinement.append(idx)
            elif validation_result.status == "fail":
                if idx not in previously_failed:
                    # give it one refinement chance
                    records_needing_refinement.append(idx)
                    new_failures.append(idx)
                    print(f"[INFO]   -> First failure, giving one refinement chance")
                else:
                    print(f"[INFO]   -> Already failed before, not retrying")

            print(
                f"[INFO]   -> status={validation_result.status}, "
                f"confidence={validation_result.actual_confidence:.3f}"
            )

        except Exception as e:
            print(f"[ERROR] Validation failed for {record_id}: {e}")
            decisions.append("fail")

    # Merge previously failed with new failures
    updated_previously_failed = list(set(previously_failed + new_failures))

    # ---- Decide overall decision based on per-record decisions ----
    if records_needing_refinement:
        overall_decision = "needs_refinement"
    elif decisions and all(d == "pass" for d in decisions):
        overall_decision = "pass"
    else:
        overall_decision = "fail"

    # Use dynamic MAX_VALIDATION_ITERATIONS from validator_module
    max_iters = getattr(validator_module, "MAX_VALIDATION_ITERATIONS", 2)
    if loop_count >= max_iters - 1:
        overall_decision = "validation_done"
        print(f"\n[INFO] MAX_VALIDATION_ITERATIONS ({max_iters}) reached. Ending validation.")

    print(f"\n[INFO] Validation complete. Overall decision: {overall_decision}")
    print(f"[INFO] Individual decisions: {decisions}")
    print(f"[INFO] Records needing refinement: {records_needing_refinement}")
    print(f"[INFO] Previously failed (won't retry): {updated_previously_failed}")

    # -----------------------------------------
    # Persist validation results to JSON
    # -----------------------------------------
    try:
        validation_output_path = output_dir / "validation_results.json"

        # Convert ValidationResult dataclasses to plain dicts
        serializable_results = {
            record_id: asdict(result)
            for record_id, result in validator_results.items()
        }

        validation_payload = {
            "overall_decision": overall_decision,
            "validation_loop": loop_count,
            "records_needing_refinement": records_needing_refinement,
            "previously_failed_indices": updated_previously_failed,
            "decisions": decisions,
            "results": serializable_results,
        }

        with validation_output_path.open("w", encoding="utf-8") as f:
            json.dump(validation_payload, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Validation results written to {validation_output_path}")
    except Exception as e:
        print(f"[WARN] Failed to write validation_results.json: {e}", file=sys.stderr)

    return {
        **state,
        "validatorDecision": overall_decision,
        "validatorResults": validator_results,
        "validationLoopCount": loop_count + 1,
        "recordsToRefine": records_needing_refinement,
        "previouslyFailedIndices": updated_previously_failed,
    }


    # -----------------------------------------
    # Persist validation results to JSON
    # -----------------------------------------
    try:
        validation_output_path = output_dir / "validation_results.json"

        # Convert ValidationResult dataclasses to plain dicts
        serializable_results = {
            record_id: asdict(result)
            for record_id, result in validator_results.items()
        }

        validation_payload = {
            "overall_decision": overall_decision,
            "validation_loop": loop_count,
            "records_needing_refinement": records_needing_refinement,
            "previously_failed_indices": updated_previously_failed,
            "decisions": decisions,
            "results": serializable_results,
        }

        with validation_output_path.open("w", encoding="utf-8") as f:
            json.dump(validation_payload, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Validation results written to {validation_output_path}")
    except Exception as e:
        print(f"[WARN] Failed to write validation_results.json: {e}", file=sys.stderr)

    
    return {
        **state,
        "validatorDecision": overall_decision,
        "validatorResults": validator_results,
        "validationLoopCount": loop_count + 1,
        "recordsToRefine": records_needing_refinement,
        "previouslyFailedIndices": updated_previously_failed,
    }

def _route_validator_results(state: PipelineState) -> str:
    """Route based on validator decision: 'pass', 'needs_refinement', or 'fail'."""
    return str(state.get("validatorDecision", "pass")).lower()

def generate_final_report_node(state: PipelineState) -> PipelineState:
    return state

# ------------------------------------------------------------
# Build the graph
# ------------------------------------------------------------

def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("search_and_process", search_and_process_node)
    graph.add_node("validator", validator_node)
    graph.add_node("final_report", generate_final_report_node)

    graph.add_edge(START, "search_and_process")
    graph.add_edge("search_and_process", "validator")

    graph.add_conditional_edges("validator", 
        _route_validator_results,
        {
            "pass": END,
            "needs_refinement": "search_and_process",
            "fail": END,
            "validation_done": END
        }
    )

    '''uncomment to use final_report node (make sure to comment out/delete the END conditional edge above)'''
    # graph.add_conditional_edges("validator", 
    #     _route_validator_results,
    #     {
    #     "pass": "final_report",
    #     "needs_refinement": "search_and_process",
    #     "fail": "final_report",
    #     "validation_done": "final_report"
    #     }
    # )

    # graph.add_edge("final_report", END)

    return graph.compile()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified MDM Search Pipeline")
    p.add_argument("--input", default="./data/query_group2.csv", help="Path to input CSV/XLSX")
    p.add_argument("--output", default="./data/", help="Directory for output")
    p.add_argument(
        "--use-dnb",
        action="store_true",
        help="Use DNB / PRIMARY_* fields along with SOURCE_* to build queries",
    )
    p.add_argument(
        "--max-validations",
        type=int,
        default=None,
        help="Max validation iterations (overrides default)",
    )
    return p.parse_args()



def _parse_comparison_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Read comparison_input_output.csv and return a list of comparison report dicts.
    
    Each dict contains a 'summary' key with input vs enriched field comparisons,
    success/message/evidence_count from the CSV.
    """
    if not csv_path.exists():
        print(f"[WARN] Comparison CSV not found at {csv_path}")
        return []
    
    comparison_reports = []
    
    # Define field mappings (input field -> enriched field)
    field_mappings = {
        "name": "enriched_canonical_name",
        "address": "enriched_address",
        "city": "enriched_city",
        "state": "enriched_state",
        "country": "enriched_country",
        "postal_code": "enriched_postal_code",
    }
    
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Build field comparisons
                field_comparisons = {}
                for input_field, enriched_field in field_mappings.items():
                    input_val = row.get(input_field, "")
                    enriched_val = row.get(enriched_field, "")
                    field_comparisons[input_field] = {
                        "input": input_val,
                        "enriched": enriched_val,
                        "changed": input_val != enriched_val and enriched_val != "",
                    }
                
                # Build the comparison report for this record
                report = {
                    "summary": {
                        "field_comparisons": field_comparisons,
                        "success": row.get("success", "").lower() == "true",
                        "message": row.get("message", ""),
                        "evidence_count": int(row.get("evidence_count") or 0),
                        "confidence": float(row.get("enriched_confidence") or 0.0),
                        "row_index": int(row.get("row_index") or 0),
                    },
                    "enriched_data": {
                        "canonical_name": row.get("enriched_canonical_name", ""),
                        "address": row.get("enriched_address", ""),
                        "city": row.get("enriched_city", ""),
                        "state": row.get("enriched_state", ""),
                        "country": row.get("enriched_country", ""),
                        "postal_code": row.get("enriched_postal_code", ""),
                        "websites": row.get("enriched_websites", ""),
                        "aka": row.get("enriched_aka", ""),
                        "lat": row.get("enriched_lat", ""),
                        "lon": row.get("enriched_lon", ""),
                    },
                    "input_data": {
                        "name": row.get("name", ""),
                        "address": row.get("address", ""),
                        "city": row.get("city", ""),
                        "state": row.get("state", ""),
                        "country": row.get("country", ""),
                        "postal_code": row.get("postal_code", ""),
                    }
                }
                comparison_reports.append(report)
                
    except Exception as e:
        print(f"[ERROR] Failed to parse comparison CSV: {e}")
        return []
    
    return comparison_reports

from pathlib import Path

def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    use_dnb: bool = False,
    max_validation_iterations: int | None = None,
) -> None:
    """
    Programmatic entrypoint for running the MDM pipeline.

    Args:
        input_path: Path to input CSV/XLSX.
        output_dir: Directory where all artifacts will be written.
        use_dnb: Whether to use DNB data when reading inputs.
        max_validation_iterations: If provided, overrides
            functions.ValidatorFunction.MAX_VALIDATION_ITERATIONS for this run.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Dynamically override max validation iterations if requested
    if max_validation_iterations is not None and max_validation_iterations > 0:
        print(f"[INFO] Setting MAX_VALIDATION_ITERATIONS = {max_validation_iterations}")
        validator_module.MAX_VALIDATION_ITERATIONS = max_validation_iterations

    print("[INFO] Starting run")
    print(f"[INFO] Input : {input_path}")
    print(f"[INFO] Output: {output_dir}")

    # -----------------------------------------
    # Step 1: Read input rows
    # -----------------------------------------
    try:
        if use_dnb:
            # read_input_rows will return (records, dnb_rows) when use_dnb=True
            records, dnb_rows = read_input_rows(str(input_path), use_dnb=True)
        else:
            records = read_input_rows(str(input_path), use_dnb=False)
            dnb_rows = []  # keep a parallel list so state always has the key
        
        print(f"[INFO] Loaded {len(records)} record(s)")

        # Save input JSON for debugging (always overwrite; this is cheap)
        input_json_path = output_dir / "input_MDM.json"
        rows_for_json = []
        for base, dnb in zip(records, dnb_rows or [None] * len(records)):
            row_dict = asdict(base)
            if dnb:           # only inject if we actually have DNB info
                row_dict["DNB"] = dnb
            rows_for_json.append(row_dict)

        with open(input_json_path, "w", encoding="utf-8") as f:
            json.dump(rows_for_json, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Saved input_MDM.json to {input_json_path}")

    except Exception as e:
        print(f"[ERROR] Failed loading input rows: {e}")
        raise

    # -----------------------------------------
    # Prepare pipeline state
    # -----------------------------------------
    initial_state: PipelineState = {
        "inputData": records,  # MUST be MDMData list
        "rawSearchResultsPath": str(output_dir / "rawSearchResults.json"),
        "processedSearchResultsPath": str(output_dir / "processedSearchResults.json"),
        "rawResults": [],
        "processedResults": {},
        "validatorResults": {},
        "validationLoopCount": 0,
        "validatorDecision": "",
        "recordsToRefine": [],
        "previouslyFailedIndices": [],
        "dnbData": dnb_rows,
        "useDnb": bool(use_dnb),
    }

    # ------------------------------------------------------------
    # Run the LangGraph pipeline
    # ------------------------------------------------------------
    graph = build_graph()
    print("[INFO] Running unified LangGraph pipeline...\n")
    final_state = graph.invoke(initial_state)
    print("\n[INFO] Pipeline completed successfully.")



# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> int:
    args = parse_args()

    try:
        run_pipeline(
            input_path=args.input,
            output_dir=args.output,
            use_dnb=bool(args.use_dnb),
            max_validation_iterations=args.max_validations,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        return 1

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
 