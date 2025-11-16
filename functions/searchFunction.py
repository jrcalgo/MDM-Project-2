# -*- coding: utf-8 -*-
#
# searchFunction.py
#
# Pipeline (no deterministic fallback for queries):
#   MDMData -> LLM-built queries -> tools (KG/Wikidata/Tavily/OSM)
#   -> LLM relevance gate (JSON mode)
#   -> LLM merge (JSON mode) with strict "evidence-only" rule + provenance
#   -> (optional) runtime enforcement: drop unprovenanced fields
#
# Env:
#   OPENAI_API_KEY           (required)
#   GOOGLE_KG_API_KEY        (optional, used in tools.py)
#   TAVILY_API_KEY           (optional, used in tools.py)
#   MDM_USER_AGENT           (default: "MDM-Enricher/1.0")
#   DEBUG_RAW                ("1" dumps raw tool payloads)
#   MERGE_DEBUG              ("1" verbose merge logs)
#   PROVENANCE_ENFORCE       ("1" drop fields without provenance; default ON)

from __future__ import annotations

import os
import json
import time
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests

from data_classes.mdm import MDMData,QueryData,OutputRow,OutputMDM

# All external tools live in a separate module
from functions.tools import (
    tavily_search,
    wikidata_search,
    wikidata_entity,
    google_kg_search,
    osm_geocode,
    reduce_kg,
    reduce_wikidata,
    reduce_tavily,
    reduce_osm,
)


# ============ Globals / Config ============

USER_AGENT = os.getenv("MDM_USER_AGENT", "MDM-Enricher/1.0")
DEBUG_RAW = os.getenv("DEBUG_RAW", "0") == "1"
MERGE_DEBUG = os.getenv("MERGE_DEBUG", "0") == "1"
PROVENANCE_ENFORCE = os.getenv("PROVENANCE_ENFORCE", "1") == "1"

# Model names (can be tweaked in one place)
QUERY_BUILDER_MODEL = "gpt-4o-mini"
GATE_MODEL = "gpt-4o-mini"
MERGE_MODEL = "gpt-4o-mini"


# ============ Utility Helpers ============

def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def debug_dump_raw(row_idx: int, stem: str, payload: Any, ext: str = "json") -> None:
    """Dump intermediate payloads for debugging."""
    ensure_dir("./debug_raw")
    path = f"./debug_raw/row{row_idx}_{stem}.{ext}"
    try:
        if ext == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(payload))
    except Exception:
        # Debug dump failure should never break the pipeline
        pass


def openai_chat(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    r = requests.post(url, headers=headers, json=payload, timeout=60)

    # >>> ADD THIS <<<
    if r.status_code != 200:
        try:
            err_json = r.json()
        except Exception:
            err_json = {"raw_text": r.text}
        # print a useful error so you can see WHY it's 400
        print(f"[OPENAI ERROR] status={r.status_code} body={json.dumps(err_json, ensure_ascii=False)}")
        r.raise_for_status()
    # <<< END ADD >>>

    data = r.json()
    return data["choices"][0]["message"]["content"]

# ============ LLM Query Builder (no fallback) ============

def llm_build_queries_source_only(m: MDMData, row_idx: int) -> QueryData:
    """
    Use an LLM to deterministically construct query strings from the raw MDMData
    based ONLY on the SOURCE_* view (current behavior).
    """
    system_prompt = (
        "You are a query builder for an MDM (Master Data Management) enrichment pipeline.\n"
        "Your job is to turn a raw company record into THREE search query strings:\n"
        "  1) q_name:        short company name text for searching knowledge graphs.\n"
        "  2) q_name_geo:    company name plus compact geographic context (city / region / country).\n"
        "  3) q_full_addr:   a single full address string optimized for geocoding APIs.\n"
        "\n"
        "RULES:\n"
        "- DO NOT change the underlying organization identity.\n"
        "- You may normalize whitespace, capitalization, and obvious abbreviations.\n"
        "- Remove clearly dummy values like '.' or '-' when constructing queries.\n"
        "- If city/state/postal/country look noisy, use the best combination you can infer from the input only.\n"
        "- Never hallucinate new locations or names that are not implied by the input.\n"
        "- For q_name_geo, keep it relatively short.\n"
        "- For q_full_addr, include as much specific address detail as is reasonable in one line.\n"
        "- Always return STRICT JSON, with keys exactly: q_name, q_name_geo, q_full_addr.\n"
    )

    user_payload = {
        "mode": "source_only",
        "input_record": asdict(m),
    }

    if MERGE_DEBUG or DEBUG_RAW:
        debug_dump_raw(row_idx, "query_builder_source_request", user_payload, ext="json")

    raw_text = openai_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        model=QUERY_BUILDER_MODEL,
        temperature=0.0,
        json_mode=True,
    )

    if MERGE_DEBUG or DEBUG_RAW:
        debug_dump_raw(row_idx, "query_builder_source_response_raw", raw_text, ext="txt")

    data = json.loads(raw_text)
    qd = QueryData(
        q_name=data.get("q_name", "").strip(),
        q_name_geo=data.get("q_name_geo", "").strip(),
        q_full_addr=data.get("q_full_addr", "").strip(),
    )
    return qd


def llm_build_queries_with_dnb(
    m: MDMData,
    dnb: Optional[Dict[str, Any]],
    row_idx: int,
) -> QueryData:
    """
    LLM query builder that sees BOTH SOURCE (MDMData) and a DNB/PRIMARY dict
    and still produces one QueryData (q_name, q_name_geo, q_full_addr).
    """
    dnb_safe: Dict[str, Any] = dnb or {}

    system_prompt = (
        "You are a query builder for an MDM enrichment pipeline.\n"
        "You receive two views of the same organization:\n"
        "- A SOURCE record (often noisy or incomplete)\n"
        "- A DNB/PRIMARY record with potentially cleaner official data\n"
        "\n"
        "Your job is to combine these into THREE search query strings:\n"
        "  1) q_name\n"
        "  2) q_name_geo\n"
        "  3) q_full_addr\n"
        "\n"
        "Treat PRIMARY/DNB fields as more authoritative when they conflict, "
        "but do not hallucinate values that are not in the input.\n"
        "\n"
        "IMPORTANT: You MUST respond with STRICT JSON (json object) only,\n"
        "with keys exactly: q_name, q_name_geo, q_full_addr.\n"
        "Do not include any extra keys or text outside the JSON."
    )

    user_payload = {
        "mode": "source_plus_dnb",
        "source_record": asdict(m),
        "dnb_record": dnb_safe,
    }

    if MERGE_DEBUG or DEBUG_RAW:
        debug_dump_raw(row_idx, "query_builder_dnb_request", user_payload, ext="json")

    try:
        raw_text = openai_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            model=QUERY_BUILDER_MODEL,
            temperature=0.0,
            json_mode=True,
        )
    except Exception as e:
        # LOG and FALL BACK to source-only behavior
        print(f"[WARN] DNB-aware query builder failed on row {row_idx}: {e}")
        return llm_build_queries_source_only(m, row_idx)

    if MERGE_DEBUG or DEBUG_RAW:
        debug_dump_raw(row_idx, "query_builder_dnb_response_raw", raw_text, ext="txt")

    data = json.loads(raw_text)
    qd = QueryData(
        q_name=data.get("q_name", "").strip(),
        q_name_geo=data.get("q_name_geo", "").strip(),
        q_full_addr=data.get("q_full_addr", "").strip(),
    )
    return qd



def build_queries(
    m: MDMData,
    row_idx: int,
    use_dnb: bool = False,
    dnb: Optional[Dict[str, Any]] = None,
) -> QueryData:
    """
    Wrapper used by the rest of the pipeline.

    - use_dnb=False  -> source-only behavior (current default)
    - use_dnb=True   -> combine SOURCE + DNB/PRIMARY dict
    """
    if use_dnb:
        return llm_build_queries_with_dnb(m, dnb, row_idx)
    return llm_build_queries_source_only(m, row_idx)


# ============ LLM Relevance Gate ============

def llm_relevance_gate(
    row_idx: int,
    input_row: MDMData,
    queries: QueryData,
    compact_tools: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decide whether the tool results plausibly match the input organization.
    Returns STRICT JSON with:
      { "relevant": bool, "why": str, "picks": { ... } }
    """
    sys_prompt = (
        "You are a precise evaluator. Return STRICT JSON only.\n"
        "Schema:\n"
        "{ \"relevant\": true|false, "
        "\"why\": \"short reason\", "
        "\"picks\": { "
        "\"kg_index\": null|0|1|2, "
        "\"wikidata_use_top\": true|false, "
        "\"tavily_indexes\": [ints], "
        "\"osm_index\": null|0|1|2 "
        "} }\n"
        "If unsure, set relevant=false and explain in 'why'."
    )

    usr = {
        "task": "Determine if tool results plausibly refer to the same organization as the input.",
        "input_company": asdict(input_row),
        "queries": asdict(queries),
        "tools_summary": compact_tools,
    }

    if MERGE_DEBUG:
        debug_dump_raw(row_idx, "gate_request", usr, ext="json")
        print(
            f'    MERGE|gate input sizes: '
            f'kg={len(compact_tools.get("google_kg") or [])}, '
            f'wd={1 if compact_tools.get("wikidata") else 0}, '
            f'tv={len(compact_tools.get("tavily") or [])}, '
            f'osm={len(compact_tools.get("osm") or [])}'
        )

    try:
        t0 = time.time()
        content = openai_chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(usr, ensure_ascii=False)},
            ],
            model=GATE_MODEL,
            temperature=0.0,
            json_mode=True,
        )
        dt_ms = int((time.time() - t0) * 1000)
        if MERGE_DEBUG:
            debug_dump_raw(row_idx, "gate_response", {"raw": content}, ext="json")
        data = json.loads(content)
        if MERGE_DEBUG:
            print(
                f"    MERGE|gate picks: relevant={data.get('relevant')} "
                f"in {dt_ms}ms | picks={data.get('picks')}"
            )
        return data
    except Exception as e:
        if MERGE_DEBUG:
            print(f"    MERGE|gate json_error: {e}")
        # For safety, treat as NOT relevant so we don't hallucinate a match.
        return {"relevant": False, "why": "llm_json_error", "picks": {}}


# ============ OutputMDM Helpers ============

def _blank_output_mdm() -> Dict[str, Any]:
    return {
        "canonical_name": "",
        "aka": [],
        "address": "",
        "city": "",
        "state": "",
        "postal_code": "",
        "country": "",
        "lat": None,
        "lon": None,
        "websites": [],
        "ids": {},
        "confidence": 0.0,
    }


def _coerce_output_mdm(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    base = _blank_output_mdm()
    missing: List[str] = []
    if not isinstance(obj, dict):
        return base, list(base.keys())
    for k in base.keys():
        if k in obj:
            base[k] = obj[k]
        else:
            missing.append(k)
    if not isinstance(base.get("aka"), list):
        base["aka"] = []
    if not isinstance(base.get("websites"), list):
        base["websites"] = []
    if not isinstance(base.get("ids"), dict):
        base["ids"] = {}
    try:
        base["confidence"] = float(base.get("confidence", 0.0) or 0.0)
    except Exception:
        base["confidence"] = 0.0
    return base, missing


def _unwrap_merge_response(data: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str, Any]:
    """
    Accepts:
      A) {"output_mdm": {...}, "evidence": [...], "notes": "...", "provenance": {...}}
      B) bare OutputMDM (then provenance/evidence empty)
    Returns: (output_mdm_dict, evidence_list, notes_str, provenance_any)
    """
    if isinstance(data, dict):
        if "output_mdm" in data:
            out = data.get("output_mdm") or {}
            ev = data.get("evidence") or []
            notes = data.get("notes", "")
            prov = data.get("provenance", {})
            return out, (ev if isinstance(ev, list) else []), notes, prov
        # Bare OutputMDM shape
        mdm_keys = {"canonical_name", "address", "city", "state", "postal_code", "country"}
        if any(k in data for k in mdm_keys):
            return data, [], "", {}
    return {}, [], "", {}


def _stringish(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _drop_unprovenanced_fields(out: Dict[str, Any], provenance: Any) -> Tuple[Dict[str, Any], List[str]]:
    """
    Drop any field that lacks provenance entries.
    Expected provenance structure (flexible, from LLM):
      {
        "canonical_name": [ { "source": "KG|Wikidata|Tavily|OSM", "text": "...", "url": "..." }, ... ],
        ...
      }
    """
    dropped: List[str] = []
    prov = provenance if isinstance(provenance, dict) else {}
    result = _blank_output_mdm()
    result.update(out or {})

    def _has_prov(field: str) -> bool:
        arr = prov.get(field)
        return isinstance(arr, list) and len(arr) > 0

    # string fields
    for field in ["canonical_name", "address", "city", "state", "postal_code", "country"]:
        if _stringish(result.get(field)) and not _has_prov(field):
            result[field] = ""
            dropped.append(field)

    # lists
    if isinstance(result.get("aka"), list) and not _has_prov("aka"):
        if result["aka"]:
            dropped.append("aka")
        result["aka"] = []

    if isinstance(result.get("websites"), list) and not _has_prov("websites"):
        if result["websites"]:
            dropped.append("websites")
        result["websites"] = []

    # ids dict
    if isinstance(result.get("ids"), dict) and not _has_prov("ids"):
        if result["ids"]:
            dropped.append("ids")
        result["ids"] = {}

    # lat/lon
    if result.get("lat") is not None and not _has_prov("lat"):
        result["lat"] = None
        dropped.append("lat")
    if result.get("lon") is not None and not _has_prov("lon"):
        result["lon"] = None
        dropped.append("lon")

    return result, dropped


# ============ LLM Merge / Extract ============

def llm_merge_extract(
    row_idx: int,
    input_row: MDMData,
    queries: QueryData,
    compact_tools: Dict[str, Any],
    picks: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str, bool, Any]:
    """
    Returns: (output_mdm, evidence, notes, ok_llm, provenance)
    No deterministic fallback: if JSON is bad, we return an empty OutputMDM and ok_llm=False.
    """
    sys_prompt = (
        "You are a careful data assembler. Return STRICT JSON only (json_object).\n"
        "RULES (very important):\n"
        "1) Only include values that are PRESENT in the tool snippets (Google KG, Wikidata, Tavily results, OSM).\n"
        "2) Do NOT infer or guess. If a field is not explicitly supported by evidence, leave it blank/empty.\n"
        "3) Do NOT copy values from the INPUT unless the exact (or trivially normalized) string also appears in at least one tool snippet.\n"
        "4) Prefer authoritative sources. If conflicting, leave blank or choose the one with the clearest citation.\n"
        "5) Provide per-field 'provenance' listing the source items and quotes/URLs you used.\n"
        "6) If nothing reliable is found, return empty fields.\n"
        "RESPONSE FORMAT (preferred):\n"
        "{\n"
        '  "output_mdm": {"canonical_name":"", "aka":[], "address":"", "city":"", "state":"", "postal_code":"", '
        '"country":"", "lat":null, "lon":null, "websites":[], "ids":{}, "confidence":0.0},\n'
        '  "evidence": [ {"source":"...","url":"...","note":"..."} ],\n'
        '  "provenance": { "field_name": [ {"source":"KG|Wikidata|Tavily|OSM","text":"...","url":"..."} ] },\n'
        '  "notes": "short rationale"\n'
        "}\n"
        "Returning the OutputMDM object directly is allowed, but provenance is strongly preferred."
    )

    usr = {
        "instruction": "Assemble OutputMDM using ONLY evidence from tools. No inference. If uncertain, leave blank.",
        "input_company": asdict(input_row),  # for reference only
        "queries": asdict(queries),
        "tools_summary": compact_tools,
        "picks": picks,
        "expected_schema": _blank_output_mdm(),
    }

    if MERGE_DEBUG:
        summary = {
            "input_company_keys": list(asdict(input_row).keys()),
            "queries": asdict(queries),
            "sizes": {
                "kg": len(compact_tools.get("google_kg") or []),
                "tavily": len(compact_tools.get("tavily") or []),
                "osm": len(compact_tools.get("osm") or []),
                "wd_has": bool(compact_tools.get("wikidata")),
            },
            "picks": picks,
        }
        debug_dump_raw(row_idx, "merge_request_summary", summary, ext="json")
        print(
            f'    MERGE|request summary: kg={summary["sizes"]["kg"]}, '
            f'wd={int(summary["sizes"]["wd_has"])}, tv={summary["sizes"]["tavily"]}, '
            f'osm={summary["sizes"]["osm"]}'
        )

    raw_text = ""
    try:
        t0 = time.time()
        raw_text = openai_chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(usr, ensure_ascii=False)},
            ],
            model=MERGE_MODEL,
            temperature=0.0,
            json_mode=True,
        )
        elapsed = int((time.time() - t0) * 1000)

        if MERGE_DEBUG:
            debug_dump_raw(row_idx, "merge_raw", raw_text, ext="txt")
            head = raw_text[:600].replace("\n", " ")
            print(
                f"    MERGE|raw response len={len(raw_text)} chars "
                f"({elapsed}ms) | head600='{head}'"
            )

        data = json.loads(raw_text)
        out_raw, ev, notes, provenance = _unwrap_merge_response(data)

        coerced, missing_keys = _coerce_output_mdm(out_raw)
        filled_keys = [
            k
            for k, v in coerced.items()
            if k not in {"aka", "websites", "ids", "confidence"} and bool(v)
        ]
        if MERGE_DEBUG:
            print(f"    MERGE|json parse: success | missing_keys={missing_keys}")
            print(
                f"    MERGE|filled keys: {filled_keys} | "
                f"websites={len(coerced.get('websites') or [])} | "
                f"ids={list((coerced.get('ids') or {}).keys())} | "
                f"conf={coerced.get('confidence')}"
            )

        if MERGE_DEBUG or DEBUG_RAW:
            debug_dump_raw(row_idx, "merge_provenance", provenance or {}, ext="json")

        return coerced, (ev if isinstance(ev, list) else []), (notes or ""), True, provenance

    except Exception as e:
        if MERGE_DEBUG:
            print(f"    MERGE|json parse/error: {e} | raw_len={len(raw_text)}")
        return _blank_output_mdm(), [], "merge_llm_json_error", False, {}


def _is_effectively_empty_output(out: Dict[str, Any]) -> bool:
    if not isinstance(out, dict) or not out:
        return True
    keys = ["canonical_name", "address", "city", "state", "postal_code", "country"]
    if any(out.get(k) for k in keys):
        return False
    if out.get("websites") or out.get("ids"):
        return False
    return True


# ============ Row processing ============

def process_row(idx: int, m: MDMData,use_dnb: bool = False,
    dnb: Optional[Dict[str, Any]] = None,) -> OutputRow:
    t0 = time.time()

    # 1) Build queries via LLM (hard dependency; no fallback)
    q = build_queries(
        m,
        row_idx=idx,
        use_dnb=use_dnb,
        dnb=dnb,
    )

    # 2) Call external tools via the tools module
    kg_status, kg_payload = google_kg_search(q.q_name_geo if q.q_name_geo else q.q_name)
    kg_reduced = reduce_kg(kg_payload)
    if DEBUG_RAW:
        debug_dump_raw(idx, "kg", kg_payload, ext="json")
    kg_hits = len(kg_reduced)
    kg_top = (kg_reduced[0].get("name") if kg_reduced else "") or ""
    print(f'  [KG] http={kg_status} | hits={kg_hits} | top="{kg_top[:60]}"')

    wd_s_status, wd_s_payload = wikidata_search(q.q_name)
    wd_e_status, wd_e_payload = (0, {})
    wd_top_label = ""
    if wd_s_status == 200 and (wd_s_payload.get("search") or []):
        wd_top_label = wd_s_payload["search"][0].get("label", "") or ""
        qid = wd_s_payload["search"][0].get("id", "")
        if qid:
            wd_e_status, wd_e_payload = wikidata_entity(qid)
    wd_reduced = reduce_wikidata(wd_s_payload, wd_e_payload)
    if DEBUG_RAW:
        debug_dump_raw(idx, "wikidata_search", wd_s_payload, ext="json")
        if wd_e_payload:
            debug_dump_raw(idx, "wikidata_entity", wd_e_payload, ext="json")
    wd_hits = len(wd_s_payload.get("search", []) or [])
    print(f'  [WD] http={wd_s_status} | hits={wd_hits} | top="{wd_top_label[:60]}"')

    tav_status, tav_payload = tavily_search(q.q_name_geo if q.q_name_geo else q.q_name)
    tav_reduced = reduce_tavily(tav_payload)
    if DEBUG_RAW:
        debug_dump_raw(idx, "tavily", tav_payload, ext="json")
    tav_hits = len(tav_reduced)
    tav_top = (tav_reduced[0].get("title") if tav_reduced else "") or ""
    print(f'  [TAVILY] http={tav_status} | hits={tav_hits} | top="{tav_top[:60]}"')

    osm_status, osm_payload = osm_geocode(q.q_full_addr)
    osm_reduced = reduce_osm(osm_payload)
    if DEBUG_RAW:
        debug_dump_raw(idx, "osm", osm_payload, ext="json")
    osm_hits = len(osm_reduced)
    osm_top = (osm_reduced[0].get("display_name") if osm_reduced else "") or ""
    print(f'  [OSM] http={osm_status} | hits={osm_hits} | top="{osm_top[:60]}"')

    # Stub; real OpenCorporates integration can go into tools.py later
    opencorporates_stub = lambda name, country: (0, {"stub": True})
    opencorporates_stub(m.name, m.country)

    compact_tools = {
        "google_kg": kg_reduced,
        "wikidata": wd_reduced,
        "tavily": tav_reduced,
        "osm": osm_reduced,
    }

    # 3) Relevance gate
    rel = llm_relevance_gate(idx, m, q, compact_tools)
    relevant = bool(rel.get("relevant", False))
    why_rel = rel.get("why", "")
    print(f"  => relevant: {'TRUE' if relevant else 'FALSE'}")

    if not relevant:
        t1 = time.time()
        return OutputRow(
            row_index=idx,
            input=asdict(m),
            queries=asdict(q),
            timing_ms=int((t1 - t0) * 1000),
            success=False,
            message=why_rel or "Not relevant.",
            public_presence=False,
            websites=[],
            evidence_count=0,
            notes=why_rel,
            enriched_mdm={},
        )

    # 4) Merge / extract via LLM
    picks = rel.get("picks", {}) if isinstance(rel.get("picks"), dict) else {}
    output_mdm, evidence, notes_merge, ok_llm, provenance = llm_merge_extract(
        idx, m, q, compact_tools, picks
    )

    enriched: Dict[str, Any] = {}
    websites: List[str] = []
    evidence_count = 0
    notes_final = notes_merge or ""

    if ok_llm and not _is_effectively_empty_output(output_mdm):
        dropped: List[str] = []
        if PROVENANCE_ENFORCE:
            output_mdm, dropped = _drop_unprovenanced_fields(output_mdm, provenance)
            if dropped:
                msg = f"dropped_unverified: {', '.join(dropped)}"
                notes_final = (notes_final + ("; " if notes_final else "") + msg).strip()
        if not _is_effectively_empty_output(output_mdm):
            enriched = output_mdm
            websites = output_mdm.get("websites") or []
            if isinstance(provenance, dict) and provenance:
                evidence_count = sum(
                    len(v) for v in provenance.values() if isinstance(v, list)
                )
            else:
                evidence_count = len(evidence)
    else:
        if MERGE_DEBUG:
            if not ok_llm:
                print("    MERGE|failed LLM merge (json).")
            else:
                print("    MERGE|empty output_mdm after coercion.")
        if not notes_final:
            notes_final = "empty_output" if ok_llm else "merge_llm_json_error"

    t1 = time.time()
    return OutputRow(
        row_index=idx,
        input=asdict(m),
        queries=asdict(q),
        timing_ms=int((t1 - t0) * 1000),
        success=bool(enriched),
        message="OK" if enriched else "Merge returned empty or failed.",
        public_presence=True,
        websites=websites,
        evidence_count=evidence_count,
        notes=notes_final,
        enriched_mdm=enriched,
    )


# ============ Batch API (used by agent.py & driver.py) ============

def run_batch(
    rows: List[MDMData],
    minimal_logging: bool = True,
    use_dnb: bool = False,
    dnb_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[OutputRow]:
    """
    Run the enrichment pipeline on a list of MDMData rows.

    - When use_dnb=False: behaves exactly as before.
    - When use_dnb=True: passes the corresponding DNB dict for each row into process_row.
    """
    outputs: List[OutputRow] = []
    dnb_rows = dnb_rows or []

    for idx, m in enumerate(rows):
        if minimal_logging:
            print(
                f'[ROW {idx}] "{(m.name or "").strip()}" '
                f'({(m.city or "").strip()}, {(m.country or "").strip()})'
            )
        dnb_for_row: Optional[Dict[str, Any]] = None
        if use_dnb and idx < len(dnb_rows):
            dnb_for_row = dnb_rows[idx]
        try:
            out_row = process_row(
                idx=idx,
                m=m,
                use_dnb=use_dnb,
                dnb=dnb_for_row,
            )
        except Exception as e:
            out_row = OutputRow(
                row_index=idx,
                input=asdict(m),
                queries={},
                timing_ms=0,
                success=False,
                message=f"Error: {e}",
                public_presence=False,
                websites=[],
                evidence_count=0,
                notes="",
                enriched_mdm={},
            )
            print(f"  !! Error processing row {idx}: {e}")
        outputs.append(out_row)

    return outputs
