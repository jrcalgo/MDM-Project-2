# Search Pipeline: How to Run & Interpret Results

This guide explains how to run the search pipeline and what each output file contains.

---

## 1) API keys

Set the following environment variables:

```bash
export OPENAI_API_KEY=...
export GOOGLE_KG_API_KEY=...
export TAVILY_API_KEY=...
```

---

## 2) Logging & enforcement flags

```bash
export MERGE_DEBUG=1      # see detailed merge/merge-gate logs (optional but helpful)
export DEBUG_RAW=1        # dump tool + LLM artifacts into ./debug_raw
export PROVENANCE_ENFORCE=1  # keep ON to drop unprovenanced fields
```

> **Note:** These flags are read at module import time. Set them **before** running `driver.py`.
If you toggle them in the same Python process, reload the module or restart the process.

---

## 3) Run the search function

```bash
python driver.py --input query_group2.xlsx --output rawSearchresults.json --formatted-output searchResults.json
```

This command will:
1. Read the input Excel (`--input`).
2. Run the full batch and write the raw output to `rawSearchresults.json`.
3. Post-process the raw output and write the condensed view to `searchResults.json`.

---

## 4) Results overview

Running the pipeline produces **two** JSON files:


### 4.1 `rawSearchresults.json`

**What it is:** the full, per-row output straight from `run_batch(...)`.

**Type:** array of objects (one per input row).

**Each object includes:**

- `row_index` — input row number  
- `input` — original input fields (`name`, `address`, `city`, `state`, `country`, `postal_code`)  
- `queries` — the constructed queries used for tools  
- `timing_ms` — processing time for this row (milliseconds)  
- `success` — boolean; `true` if the merge produced non-empty enrichment after enforcement  
- `message` — short status text  
- `public_presence` — whether any tool returned hits  
- `websites` — any websites found (if any)  
- `evidence_count` — count of provenance/evidence items  
- `notes` — merge notes (e.g., dropped fields, reasons)  
- `enriched_mdm` — the full enrichment object (may be empty if not enough evidence)

**`enriched_mdm` keys:**  
`canonical_name`, `aka[]`, `address`, `city`, `state`, `postal_code`, `country`, `lat`, `lon`, `websites[]`, `ids{}`, `confidence`

**Example (abridged):**
```json
[
  {
    "row_index": 3,
    "input": { "name": "Bas Engineering Limited", "...": "..." },
    "queries": { "q_name": "Bas Engineering Limited", "...": "..." },
    "timing_ms": 26454,
    "success": true,
    "message": "OK",
    "public_presence": true,
    "websites": ["http://www.bashk.com/template?series=38"],
    "evidence_count": 8,
    "notes": "Data assembled from Tavily sources...",
    "enriched_mdm": {
      "canonical_name": "BAS Engineering Limited",
      "aka": [],
      "address": "Room 708, Paramount Building, 12 Ka Yip Street",
      "city": "Chai Wan",
      "state": "HONG KONG",
      "postal_code": "",
      "country": "Hong Kong",
      "lat": null,
      "lon": null,
      "websites": ["http://www.bashk.com/template?series=38"],
      "ids": {},
      "confidence": 0.0
    }
  }
]
```

> If `MERGE_DEBUG=1` or `DEBUG_RAW=1`, auxiliary artifacts are also written under `./debug_raw/`.


### 4.2 `searchResults.json`

**What it is:** a condensed, post-processed view for downstream use.

**Type:** object with a single key `searchResults` → array (one item per input row).

**Per-row behavior:**

- If search **succeeded** and `enriched_mdm` has meaningful information → include **all fields** (initialized to empty strings if missing):
  ```json
  {
    "name": "<from enriched_mdm.canonical_name or "">",
    "address": "<from enriched_mdm.address or "">",
    "city": "<from enriched_mdm.city or "">",
    "state": "<from enriched_mdm.state or "">",
    "country": "<from enriched_mdm.country or "">",
    "postal_code": "<from enriched_mdm.postal_code or "">",
    "search": true
  }
  ```

- If search did **not** succeed (or enrichment is effectively empty) → just:
  ```json
  { "search": false }
  ```

**Example:**
```json
{
  "searchResults": [
    {
      "name": "BAS Engineering Limited",
      "address": "Room 708, Paramount Building, 12 Ka Yip Street",
      "city": "Chai Wan",
      "state": "HONG KONG",
      "country": "Hong Kong",
      "postal_code": "",
      "search": true
    },
    { "search": false }
  ]
}
```
