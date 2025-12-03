# MDM-Project-2

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables** (use the supplied `.env` file or export):
   ```bash
   export OPENAI_API_KEY=your_key_here
   export GROQ_API_KEY=your_key_here
   export GOOGLE_KG_API_KEY=your_key_here  # optional
   export TAVILY_API_KEY=your_key_here      # optional
   export MERGE_DEBUG=1             # see detailed merge logs (optional but helpful)
   export DEBUG_RAW=1               # dump tool + LLM artifacts into ./debug_raw
   export PROVENANCE_ENFORCE=1      # keep this ON to drop unprovenanced fields
   ```
Requires Python 3.12+.

3. **Run the pipeline**

```bash
python agent.py --output ./data/
```

This command will:
1. Read the input Excel/csv file (`--input`: Defaults to ./data/query_group2.csv).
2. Produce a json which object for processing as `input_MDM.json`.
3. Run the full batch and write the raw output to `rawSearchResults.json`.
4. Post-process the raw output and write the condensed view to `processedSearchResults.json`.
5. Compares `input_MDM.json` to `processedSearchResults.json`, yielding `comparison_input_output.csv`.
6. Validates/scrutinizes the results of `comparison_input_output.csv` and provides feedback.
7. If feedback indicates overall need for refinement or failure, search again using recommendations.
8. If feedback indicates pass or validation loop completion, continue to final analysis report.
9. Generation and output of report document.
