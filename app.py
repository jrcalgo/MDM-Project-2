# app.py

from __future__ import annotations

from pathlib import Path
import io
import csv
import uuid
import json

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    abort,
)
from werkzeug.utils import secure_filename
import os
import typing as t
import requests

from agent import run_pipeline
from utils.report_builder import build_merged_report, REPORT_COLUMNS

# -----------------------------------------------------------------------------
# Flask app configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
RUNS_FOLDER = BASE_DIR / "runs"

UPLOAD_FOLDER.mkdir(exist_ok=True)
RUNS_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}


def allowed_file(filename: str) -> bool:
    """Return True if the uploaded filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__, static_folder="static", template_folder="templates")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def index():
    """Landing page with upload form."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle file upload, run the MDM pipeline, then redirect to the report view."""
    if "file" not in request.files:
        return "No file part in request", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    if not allowed_file(file.filename):
        return "Unsupported file type. Please upload CSV or Excel.", 400

    filename = secure_filename(file.filename)
    run_id = uuid.uuid4().hex
    run_dir = RUNS_FOLDER / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / filename
    file.save(str(input_path))

    # Checkbox for DNB / PRIMARY_* fields
    use_dnb = bool(request.form.get("use_dnb"))

    # Optional: max validation iterations (from form)
    max_validations_raw = (request.form.get("max_validations") or "").strip()
    max_validations = None
    if max_validations_raw:
        try:
            mv = int(max_validations_raw)
            if mv > 0:
                max_validations = mv
        except ValueError:
            # Ignore invalid values and fall back to default
            max_validations = None

    try:
        run_pipeline(
            input_path=input_path,
            output_dir=run_dir,
            use_dnb=use_dnb,
            max_validation_iterations=max_validations,
        )
    except Exception as e:  # pragma: no cover - simple error surface
        # In a real app you might show a nicer error page or log this.
        return f"Pipeline failed: {e}", 500

    return redirect(url_for("report", run_id=run_id))


@app.route("/report/<run_id>", methods=["GET"])
def report(run_id: str):
    """
    Render the HTML report for a given run.

    The view combines the merged CSV-style report with the validation payload
    written by the validator into ``validation_results.json``.
    """
    run_dir = RUNS_FOLDER / run_id
    if not run_dir.exists():
        abort(404, description="Run not found")

    try:
        rows = build_merged_report(run_dir)
    except Exception as e:
        return f"Failed to build report: {e}", 500

    # Load validation_results.json (plural) written by the pipeline
    validation_path = run_dir / "validation_results.json"
    report_data = None
    summary = None

    if validation_path.exists():
        with validation_path.open("r", encoding="utf-8") as f:
            report_data = json.load(f)

        # Build a small status summary for the header chips
        results = (report_data or {}).get("results", {}) or {}
        total = len(results)
        fails = sum(1 for r in results.values() if r.get("status") == "fail")
        needs_refinement = sum(
            1 for r in results.values() if r.get("status") == "needs_refinement"
        )
        passes = sum(1 for r in results.values() if r.get("status") == "pass")

        summary = {
            "total": total,
            "fails": fails,
            "needs_refinement": needs_refinement,
            "passes": passes,
        }

    return render_template(
        "report.html",
        run_id=run_id,
        columns=REPORT_COLUMNS,
        rows=rows,
        report=report_data,
        summary=summary,
    )


def _call_openai_chat(system: str, user: str) -> str:
    """Call OpenAI chat completions API and return assistant text.

    Requires `OPENAI_API_KEY` environment variable. If not set or the call
    fails, raise an Exception.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error: {resp.status_code} {resp.text}")

    data = resp.json()
    # Navigate to the assistant content
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Unexpected OpenAI response: {data}") from exc


@app.route("/report/<run_id>/explain", methods=["GET"])
def explain_record(run_id: str):
    """Return an LLM-generated paragraph explaining the merged row compared
    to the original input values for a single record.

    Query param: `record_id` (the id shown in the validation results keys).
    """
    record_id = request.args.get("record_id")
    if not record_id:
        return {"error": "missing record_id"}, 400

    run_dir = RUNS_FOLDER / run_id
    if not run_dir.exists():
        return {"error": "run not found"}, 404

    # Try cache first
    cache_path = run_dir / "explanations.json"
    cache: t.Dict[str, str] = {}
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                cache = json.load(f) or {}
        except Exception:
            cache = {}

    if record_id in cache:
        return {"explanation": cache[record_id]}

    # Build merged rows and find the requested row
    try:
        rows = build_merged_report(run_dir)
    except Exception as e:
        return {"error": f"failed to build rows: {e}"}, 500

    found = None
    for r in rows:
        idx = r.get("ROW_INDEX")
        input_name = r.get("INPUT_NAME") or ""
        key_by_idx = f"record_{idx}"
        if record_id == input_name or record_id == key_by_idx:
            found = r
            break

    if not found:
        # fallback: try to match by INPUT_NAME equals record_id
        for r in rows:
            if r.get("INPUT_NAME") == record_id:
                found = r
                break

    if not found:
        return {"error": "record not found in merged rows"}, 404

    # Build a compact prompt describing input/enriched/validation fields
    # Use short label: value pairs so LLM can compare easily.
    lines = []
    lines.append(f"Record identifier: {record_id}")
    lines.append("\nInput fields:")
    for k in ("INPUT_NAME", "INPUT_ADDRESS", "INPUT_CITY", "INPUT_STATE", "INPUT_POSTAL_CODE", "INPUT_COUNTRY"):
        lines.append(f"- {k}: {found.get(k, '')}")

    lines.append("\nEnriched/search fields:")
    for k in ("SEARCH_CANONICAL_NAME", "SEARCH_AKA", "SEARCH_ADDRESS", "SEARCH_CITY", "SEARCH_STATE", "SEARCH_POSTAL_CODE", "SEARCH_COUNTRY", "SEARCH_WEBSITES", "SEARCH_IDS", "SEARCH_CONFIDENCE"):
        lines.append(f"- {k}: {found.get(k, '')}")

    lines.append("\nValidation fields:")
    for k in ("VALIDATION_STATUS", "CONFIDENCE_MEETS_THRESHOLD", "THRESHOLD_USED", "ACTUAL_CONFIDENCE", "PROVENANCE_QUALITY_SCORE", "VALIDATION_NOTES"):
        lines.append(f"- {k}: {found.get(k, '')}")

    prompt_user = (
        "Please write one concise informative paragraph (2-4 sentences) that: "
        "(1) summarizes how the validation result compares to the original input and enriched fields, "
        "(2) highlights any important differences, discrepancies, or noteworthy matches, and "
        "(3) mentions the most relevant fields the reviewer should check. "
        "Use factual, neutral language and avoid fabricating facts.\n\n" + "\n".join(lines)
    )

    system_msg = (
        "You are an assistant that converts structured MDM validation rows into a human-readable paragraph for reviewers. "
        "Keep the text factual and concise. If data is missing, state that it is missing. "
    )

    # Call OpenAI if configured
    try:
        explanation = _call_openai_chat(system_msg, prompt_user)
    except Exception as e:
        # If LLM is not configured/available, return a helpful fallback short summary
        explanation = (
            "Summary unavailable (LLM not configured). Fallback: "
            + f"Status={found.get('VALIDATION_STATUS')}; Confidence={found.get('ACTUAL_CONFIDENCE')}; "
            + f"Threshold={found.get('THRESHOLD_USED')}; Notes={found.get('VALIDATION_NOTES') or 'none'}"
        )

    # Cache and return
    try:
        cache[record_id] = explanation
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        # ignore cache write errors
        pass

    return {"explanation": explanation}


@app.route("/report/<run_id>/download", methods=["GET"])
def download_report(run_id: str):
    """
    Stream the merged report as a CSV download.
    """
    run_dir = RUNS_FOLDER / run_id
    if not run_dir.exists():
        abort(404, description="Run not found")

    try:
        rows = build_merged_report(run_dir)
    except Exception as e:
        return f"Failed to build report: {e}", 500

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(REPORT_COLUMNS)
    for row in rows:
        writer.writerow([row.get(col, "") for col in REPORT_COLUMNS])

    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"mdm_report_{run_id}.csv",
    )


if __name__ == "__main__":
    # For local development; in production use a WSGI server (gunicorn/uwsgi, etc.)
    app.run(debug=True, host="0.0.0.0", port=5001)
