# MDM Enrichment & Validation Pipeline

This project runs an end-to-end **MDM (Master Data Management) enrichment and validation pipeline** backed by LLM tools, and exposes:

- A **CLI** for batch processing CSV/XLSX input files.
- A **Flask web UI** for uploading files, running the pipeline, and exploring/downloading the final report.

The pipeline reads customer/organization records, calls search/enrichment tools, validates the results using an LLM-driven validation loop, and produces both a **merged CSV report** and a structured **`validation_results.json`** payload per run.

---

## Features

- 🔎 **Search & enrichment** of MDM records via multiple tools and LLMs.
- ✅ **Validation loop** that checks confidence thresholds, provenance, and issues; supports refinement iterations.
- 📊 **Merged report generation** combining original input, enriched fields, DNB fields (optional), and validation status.
- 🌐 **Web UI (Flask)**:
  - Upload CSV/XLSX files.
  - Configure use of DNB/PRIMARY fields.
  - Optionally override max validation iterations.
  - View per-record validation summary.
  - Download the final CSV report.
- 🧪 **CLI mode** for running the full pipeline from the terminal.

---

## Project structure

```text
.
├── app.py                       # Flask web application (upload + report)
├── agent.py                     # Orchestrates the LangGraph / LLM pipeline
├── data_classes/
│   └── mdm.py                   # Dataclasses for input and output records
├── functions/
│   ├── __init__.py
│   ├── ComparisonFunction.py    # Compares original vs enriched records
│   ├── Generate_CSV.py          # Builds comparison_input_output.csv
│   ├── searchFunction.py        # Runs search/enrichment tools
│   ├── tools.py                 # Tool wiring for the pipeline
│   ├── ValidatorFunction.py     # LLM-driven validation logic
│   └── processSearchResults.py  # Condenses raw search results
├── utils/
│   ├── csv2json.py              # Helper to convert CSV → JSON
│   ├── read_input_records.py    # Reads Excel/CSV inputs into MDMData
│   └── report_builder.py        # Builds merged report rows for UI/CSV
├── templates/
│   ├── base.html                # Shared layout (Gradio-like light theme)
│   ├── index.html               # Upload form + pipeline overview
│   └── report.html              # Run summary, validation details, data table
├── static/
│   └── style.css                # Global styling for the web UI
├── data/
│   └── query_group2.csv         # Example input (optional)
├── uploads/                     # Created at runtime; stores uploaded files
├── runs/                        # Created at runtime; one subfolder per run
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

> Note: `uploads/` and `runs/` are created automatically by `app.py` if they do not exist.

---

## Installation

1. **Clone** the repository and enter the project directory.

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

The pipeline can use multiple external providers and tools. At minimum, you’ll need a valid LLM provider key (for example, OpenAI or Groq).

Environment variables commonly used:

```bash
export OPENAI_API_KEY=your_openai_key_here
export GROQ_API_KEY=your_groq_key_here
export GOOGLE_KG_API_KEY=your_google_kg_key_here   # optional
export TAVILY_API_KEY=your_tavily_key_here         # optional
```

You can either:

- Set these in your shell, or  
- Place them in a `.env` file and use `python-dotenv` (already supported in the code).

---

## Running the CLI pipeline

The CLI entry point is `agent.py`. Typical usage:

```bash
python agent.py \
  --input ./data/query_group2.csv \
  --output ./data/ \
  --use-dnb \
  --max-validations 3
```

Arguments (from `agent.py`):

- `--input` (str): Path to the input CSV/XLSX file  
  _Default_: `./data/query_group2.csv`
- `--output` (str): Directory where intermediate and final outputs are written  
  _Default_: `./data/`
- `--use-dnb`: If provided, uses DNB / PRIMARY_* fields along with SOURCE_* fields to build queries.
- `--max-validations` (int): Optional override for the maximum validation iterations.

The CLI will:

1. Read the input Excel/CSV file.
2. Generate `input_MDM.json` with normalized records.
3. Run the search/enrichment step and write `rawSearchResults.json`.
4. Condense search output to `processedSearchResults.json`.
5. Compare input vs enriched data to create `comparison_input_output.csv`.
6. Run validation and write `validation_results.json` (overall decision + per-record results).
7. Optionally perform refinement loops if the decision indicates `needs_refinement`.
8. Produce the final analysis artifacts for downstream consumption.

---

## Running the web UI

The Flask web app lives in `app.py`.

1. Make sure your environment variables and virtual environment are set.
2. Start the server:

   ```bash
   python app.py
   ```

3. Open your browser and navigate to:

   ```text
   http://localhost:5001/
   ```

### Web UI workflow

1. **Upload** a CSV/XLSX file from the landing page.
2. Optionally:
   - Check “Use DNB / PRIMARY_* fields” if your file includes those columns.
   - Set “Max validation iterations” to override the default per-run.
3. Click **Run pipeline**.
4. Once the pipeline completes, you are redirected to the **report** page for that run:
   - A **Validation summary** card shows total records, pass/fail/needs-refinement counts, overall decision, and validation loop count.
   - A **Per-record validation table** shows status, confidence, threshold, provenance score, iterations, and notes, based on `validation_results.json`.
   - A **Merged MDM report table** mirrors the downloadable CSV, including all columns defined in `REPORT_COLUMNS`.
   - A small **filter box** lets you quickly search within the merged table.
5. Click **Download CSV** to download the merged report for the current run.

Each run’s artifacts (uploads, JSON files, CSVs) are stored under:

```text
runs/<run_id>/
```

---

## Data outputs per run

For each web/CLI run, you should expect a folder like:

```text
runs/<run_id>/
├── <original_uploaded_file>.csv / .xlsx
├── input_MDM.json
├── rawSearchResults.json
├── processedSearchResults.json
├── comparison_input_output.csv
├── validation_results.json
└── (any additional debug artifacts configured in the pipeline)
```

The web UI reads:

- `validation_results.json` for validation summary and per-record status.
- The merged report rows via `utils.report_builder.build_merged_report()` for the table and CSV download.

---

## Development notes

- The Flask app is intentionally thin: it delegates all heavy lifting to `agent.run_pipeline` and `utils.report_builder`.
- `validation_results.json` contains:
  - `overall_decision`
  - `validation_loop`
  - `records_needing_refinement`
  - `previously_failed_indices`
  - `decisions`
  - `results` (dict of per-record `ValidationResult` payloads)
- `report_builder.REPORT_COLUMNS` controls the column order for both the HTML table and the CSV download.

If you change the validation payload or report columns, make sure to update:

- `utils/report_builder.py` for the merged rows, and
- `templates/report.html` if you want to surface new fields in the UI.

---

## License

Internal / project use only (update this section with the appropriate license if you plan to publish the repository).
