import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, TypedDict

# IMPORTANT: import the actual functions, not the modules
from functions.searchFunction import read_input_rows, run_batch
from functions.processSearchResults import process_results
from functions.ComparisonFunction import ComparisonFunction
from functions.Generate_CSV import GenerateCSVTool

comparisonTool = ComparisonFunction().compare_records

from langgraph.graph import StateGraph, END


# ------------------------------------------------------------
# State definition
# ------------------------------------------------------------

class PipelineState(TypedDict):
    inputData: List[Dict[str, Any]]
    rawResults: List[Dict[str, Any]]
    processedResults: Dict[str, Any]
    rawSearchResultsPath: str
    processedSearchResultsPath: str


# ------------------------------------------------------------
# Unified node (search + process)
# ------------------------------------------------------------

def search_and_process_node(state: PipelineState) -> PipelineState:

    records = state["inputData"]
    raw_path = state["rawSearchResultsPath"]
    processed_path = state["processedSearchResultsPath"]

    print("[INFO] Starting unified search + process node")
    print(f"[INFO] Processing {len(records)} record(s)...\n")

    # -----------------------------------------
    # Step 1: Run search
    # -----------------------------------------
    try:
        raw_results = run_batch(records)
    except Exception as e:
        print(f"[ERROR] searchFunction.run_batch failed: {e}", file=sys.stderr)
        raise

    # -----------------------------------------
    # Step 2: Write raw results
    # -----------------------------------------
    try:
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Raw search results written to {raw_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write raw results: {e}")
        raise

    # -----------------------------------------
    # Step 3: Process results
    # -----------------------------------------
    try:
        processed = process_results(raw_path)
    except Exception as e:
        print(f"[ERROR] process_results failed: {e}")
        raise

    # -----------------------------------------
    # Step 4: Write processed results
    # -----------------------------------------
    try:
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Processed results written to {processed_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write processed results: {e}")
        raise

    # -----------------------------------------
    # Step 5: Compare Input vs Processed Results
    # -----------------------------------------

    try:
        comparison = comparisonTool(records, processed.get("searchResults", []))
        print("\n[INFO] Comparison of Input vs Processed Results:")
        print(f"Comparison Results: {json.dumps(comparison, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"[ERROR] ComparisonFunction failed: {e}")
        raise

    # -----------------------------------------
    # Step 6: Return updated state
    # ----------------------------------------- 

    try:
        csv_tool = GenerateCSVTool()
        # processed is expected to be a dict with key "searchResults" -> list aligned with records
        processed_list = processed.get("searchResults") if isinstance(processed, dict) else None
        if processed_list is None:
            # if processed format is unexpected, just wrap processed to keep CSV call tolerant
            processed_list = processed

        # Build CSV path next to processed JSON output
        csv_path = Path(processed_path).with_name("comparison_input_output.csv")
        csv_tool.run(records, processed_list, out_path=str(csv_path))
        print(f"[INFO] Comparison CSV written to {csv_path}")
    except Exception as e:
        # non-fatal: pipeline continues; but log the issue
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
        records = read_input_rows(str(input_path))
        print(f"[INFO] Loaded {len(records)} record(s)")

        # Save input JSON for debugging
        input_json_path = output_dir / "input_MDM.json"
        records_dict = [asdict(r) for r in records]
        with open(input_json_path, "w", encoding="utf-8") as f:
            json.dump(records_dict, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Saved input_MDM.json to {input_json_path}")


    except Exception as e:
        print(f"[ERROR] Failed loading input rows: {e}")
        return 1

    # -----------------------------------------
    # Prepare pipeline state
    # -----------------------------------------
    # Initial pipeline state
    initial_state: PipelineState = {
        "inputData": records,  # MUST be MDMData list
        "rawSearchResultsPath": str(output_dir / "rawSearchResults.json"),
        "processedSearchResultsPath": str(output_dir / "processedSearchResults.json"),
        "rawResults": [],
        "processedResults": {},
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
