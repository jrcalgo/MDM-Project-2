# driver.py
# Entry point for the full pipeline.
# Usage:
#   python driver.py --input /path/to/input.xlsx --output results.json --formatted-output formattedResults.json

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

# Import your project logic
from functions import searchFunction # exposes: read_input_rows(path), run_batch(records, minimal_logging=True)
from functions import processSearchResults # exposes: process_results(results_json_path) -> dict

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MDM verification pipeline driver")
    p.add_argument("--input", required=True, help="Path to input file (CSV/XLSX)")
    p.add_argument("--output", default="rawSearchResults.json", help="Where to write the raw results JSON")
    p.add_argument("--formatted-output", default="searchResults.json",
                   help="Where to write the processed/condensed results JSON")
    p.add_argument("--no-minimal-logging", action="store_true",
                   help="Disable minimal logging flag when calling run_batch")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    formatted_path = Path(args.formatted_output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 2

    start_ts = time.time()
    print("[INFO] Starting run")
    print(f"[INFO] Input : {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Formatted Output: {formatted_path}")

    try:
        # 1) Build MDMData records
        records = searchFunction.read_input_rows(str(input_path))
        total = len(records) if hasattr(records, "__len__") else None
        if total is not None:
            print(f"[INFO] Loaded {total} record(s)")

        # 2) Run the full batch
        minimal_logging = not args.no_minimal_logging
        results = searchFunction.run_batch(records, minimal_logging=minimal_logging)

        # 3) Write results EXACTLY as returned by run_batch
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Wrote raw results: {output_path}")

        # 4) Process results into the condensed schema and write it too
        processed = processSearchResults.process_results(str(output_path))
        formatted_path.parent.mkdir(parents=True, exist_ok=True)
        with formatted_path.open("w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Wrote formatted results: {formatted_path}")

        dur = time.time() - start_ts
        print(f"[INFO] Done in {dur:.2f}s")
        return 0

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
