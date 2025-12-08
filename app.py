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
