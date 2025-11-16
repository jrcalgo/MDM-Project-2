import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, TypedDict

from data_classes.mdm import MDMData
from functions.searchFunction import run_batch
from utils.read_input_records import read_input_rows
from functions.processSearchResults import process_results
from functions.ComparisonFunction import ComparisonFunction
from functions.Generate_CSV import GenerateCSVTool

from langgraph.graph import StateGraph, END

comparisonTool = ComparisonFunction().compare_records


# ------------------------------------------------------------
# State definition
# ------------------------------------------------------------

class PipelineState(TypedDict):
    inputData: List[MDMData]          # list of input records as dataclasses
    rawResults: List[Dict[str, Any]]  # list of asdict(OutputRow)
    processedResults: Dict[str, Any]  # processedSearchResults.json structure
    rawSearchResultsPath: str
    processedSearchResultsPath: str

    dnbData: List[Dict[str, Any]]   # parallel list of DNB dicts (can be empty)
    useDnb: bool                    # whether we are using DNB in this run


# ------------------------------------------------------------
# Unified node (search + process)
# ------------------------------------------------------------

def search_and_process_node(state: PipelineState) -> PipelineState:

    records = state["inputData"]
    use_dnb = state.get("useDnb", False)
    dnb_data = state.get("dnbData", [])
    raw_path = state["rawSearchResultsPath"]
    processed_path = state["processedSearchResultsPath"]

    raw_path_obj = Path(raw_path)
    processed_path_obj = Path(processed_path)

    print("[INFO] Starting unified search + process node")
    print(f"[INFO] Processing {len(records)} record(s)...\n")

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
    # Step 3: Get processed results (post-processing, with skip logic)
    # -----------------------------------------
    if processed_path_obj.exists():
        print(f"[INFO] Found existing processed results at {processed_path_obj}. Skipping process_results step.")
        try:
            with processed_path_obj.open("r", encoding="utf-8") as f:
                processed = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read existing processed results from {processed_path_obj}: {e}", file=sys.stderr)
            raise
    else:
        print("[INFO] No existing processed results found. Running process_results...")
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


# ------------------------------------------------------------
# Build the graph
# ------------------------------------------------------------

def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("search_and_process", search_and_process_node)
    graph.set_entry_point("search_and_process")
    graph.add_edge("search_and_process", END)
    return graph.compile()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified MDM Search Pipeline")
    p.add_argument("--input", required=True, help="Path to input CSV/XLSX")
    p.add_argument("--output", default="./data/", help="Directory for output")
    p.add_argument(
    "--use-dnb",
    action="store_true",
    help="Use DNB / PRIMARY_* fields along with SOURCE_* to build queries",
)
    return p.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 2

    print("[INFO] Starting run")
    print(f"[INFO] Input : {input_path}")

    # -----------------------------------------
    # Step 1: Read input rows
    # -----------------------------------------
    try:
        if args.use_dnb:
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
        return 1

    # -----------------------------------------
    # Prepare pipeline state
    # -----------------------------------------
    initial_state: PipelineState = {
        "inputData": records,  # MUST be MDMData list
        "rawSearchResultsPath": str(output_dir / "rawSearchResults.json"),
        "processedSearchResultsPath": str(output_dir / "processedSearchResults.json"),
        "rawResults": [],
        "processedResults": {},
        "dnbData": dnb_rows,
        "useDnb": bool(args.use_dnb),
    }

    # ------------------------------------------------------------
    # Run the LangGraph pipeline
    # ------------------------------------------------------------

    try:
        graph = build_graph()
        print("[INFO] Running unified LangGraph pipeline...\n")
        final_state = graph.invoke(initial_state)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        return 1

    print("\n[INFO] Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
