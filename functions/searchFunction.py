# -*- coding: utf-8 -*-
# 
#
# Pipeline (no deterministic fallback):
#   Excel -> build queries (input untouched) -> tools (KG/Wikidata/Tavily/OSM)
#   -> LLM relevance gate (JSON mode)
#   -> LLM merge (JSON mode) with strict "evidence-only" rule + provenance
#   -> (optional) runtime enforcement: drop fields without provenance
#   -> write results
#
# Env:
#   OPENAI_API_KEY           (required)
#   GOOGLE_KG_API_KEY        (optional)
#   TAVILY_API_KEY           (optional)
#   MDM_INPUT_XLSX           (default: query_group2.xlsx)
#   MDM_OUT_JSON             (default: mdm_results.json)
#   MDM_USER_AGENT           (default: "MDM-Enricher/1.0")
#   DEBUG_RAW                ("1" dumps raw tool payloads)
#   MERGE_DEBUG              ("1" verbose merge logs)
#   MDM_MAX_ROWS             (int; optional cap)
#   PROVENANCE_ENFORCE       ("1" drop fields without provenance; default ON)

from __future__ import annotations

import os
import re
import json
import time
import unicodedata
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd

# ============ Models ============

@dataclass
class MDMData:
    name: str
    address: str
    city: str
    state: str
    country: str
    postal_code: str

@dataclass
class QueryData:
    q_name: str
    q_name_geo: str
    q_full_addr: str

@dataclass
class OutputRow:
    row_index: int
    input: Dict[str, Any]
    queries: Dict[str, Any]
    timing_ms: int
    success: bool
    message: str
    public_presence: bool
    websites: List[str]
    evidence_count: int
    notes: str
    enriched_mdm: Dict[str, Any]

# ============ Globals / Utils ============

USER_AGENT = os.getenv("MDM_USER_AGENT", "MDM-Enricher/1.0")
DEBUG_RAW = os.getenv("DEBUG_RAW", "0") == "1"
MERGE_DEBUG = os.getenv("MERGE_DEBUG", "0") == "1"
PROVENANCE_ENFORCE = os.getenv("PROVENANCE_ENFORCE", "1") == "1"

def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def debug_dump_raw(row_idx: int, stem: str, payload: Any, ext: str = "json") -> None:
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
        pass

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def country_label(country: str) -> str:
    if not country:
        return ""
    c = normalize_text(country).casefold()
    mapping = {
        "cn": "China",
        "china": "China",
        "hk": "Hong Kong",
        "hong kong": "Hong Kong",
        "mo": "Macao",
        "tw": "Taiwan",
        "us": "United States",
        "usa": "United States",
        "uk": "United Kingdom",
    }
    return mapping.get(c, country)

def treat_hong_kong(country: str, city: str) -> Tuple[str, str]:
    if normalize_text(country).casefold() in {"cn", "china"} and "hong kong" in normalize_text(city).casefold():
        return "Hong Kong", city
    return country_label(country), city

def build_full_addr(address: str, city: str, state: str, postal: str, country: str) -> str:
    parts = []
    if address:
        parts.append(address)
    mid = " ".join([x for x in [city, state, postal] if x and x != "-"]).strip()
    if mid:
        parts.append(mid)
    if country:
        parts.append(country)
    return ", ".join(parts)

# ============ OpenAI helper (JSON mode) ============

def openai_chat(messages: List[Dict[str, str]],
                model: str = "gpt-4o-mini",
                temperature: float = 0.0,
                max_tokens: Optional[int] = None,
                json_mode: bool = False) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ============ Tool calls ============

def tavily_search(query: str) -> Tuple[int, Dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return (0, {"error": "no_api_key"})
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"query": query, "search_depth": "basic", "max_results": 5}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=45)
        return r.status_code, (r.json() if r.content else {})
    except Exception as e:
        return (-1, {"error": str(e)})

def wikidata_search(name: str) -> Tuple[int, Dict[str, Any]]:
    try:
        params = {"action": "wbsearchentities", "format": "json", "language": "en", "search": name, "limit": 3}
        r = requests.get("https://www.wikidata.org/w/api.php", params=params, timeout=45,
                         headers={"User-Agent": USER_AGENT})
        return r.status_code, (r.json() if r.content else {})
    except Exception as e:
        return (-1, {"error": str(e)})

def wikidata_entity(qid: str) -> Tuple[int, Dict[str, Any]]:
    try:
        params = {"action": "wbgetentities", "format": "json", "ids": qid, "props": "sitelinks|labels|aliases|descriptions|claims"}
        r = requests.get("https://www.wikidata.org/w/api.php", params=params, timeout=45,
                         headers={"User-Agent": USER_AGENT})
        return r.status_code, (r.json() if r.content else {})
    except Exception as e:
        return (-1, {"error": str(e)})

def google_kg_search(query: str) -> Tuple[int, Dict[str, Any]]:
    api_key = os.getenv("GOOGLE_KG_API_KEY", "")
    if not api_key:
        return (0, {"error": "no_api_key"})
    try:
        params = {"query": query, "limit": 3, "indent": True, "key": api_key}
        r = requests.get("https://kgsearch.googleapis.com/v1/entities:search",
                         params=params, timeout=45, headers={"User-Agent": USER_AGENT})
        return r.status_code, (r.json() if r.content else {})
    except Exception as e:
        return (-1, {"error": str(e)})

def osm_geocode(addr: str) -> Tuple[int, Any]:
    try:
        params = {"q": addr, "format": "json", "limit": 3, "addressdetails": 1}
        r = requests.get("https://nominatim.openstreetmap.org/search", params=params, timeout=45,
                         headers={"User-Agent": USER_AGENT})
        return r.status_code, (r.json() if r.content else [])
    except Exception as e:
        return (-1, {"error": str(e)})

def opencorporates_stub(name: str, country: str) -> Tuple[int, Dict[str, Any]]:
    return (0, {"stub": True})

# ============ Reducers for LLM prompts ============

def reduce_kg(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = payload.get("itemListElement", []) or []
    out = []
    for it in items[:3]:
        res = it.get("result", {}) or {}
        out.append({
            "name": res.get("name"),
            "type": res.get("@type"),
            "description": (res.get("description") or ""),
            "detailed_url": (res.get("detailedDescription", {}) or {}).get("url"),
            "kg_id": res.get("@id")
        })
    return out

def reduce_wikidata(search_payload: Dict[str, Any], entity_payload: Dict[str, Any]) -> Dict[str, Any]:
    search = search_payload.get("search", []) or []
    top = None
    if search:
        top = {
            "id": search[0].get("id"),
            "label": search[0].get("label"),
            "description": search[0].get("description")
        }
    sitelinks = {}
    descriptions = {}
    if entity_payload:
        ents = entity_payload.get("entities", {}) or {}
        if top and top.get("id") in ents:
            e = ents[top["id"]]
            sl = e.get("sitelinks", {}) or {}
            for k, v in sl.items():
                if "wiki" in k and isinstance(v, dict) and v.get("url"):
                    sitelinks[k] = v.get("url")
            desc = e.get("descriptions", {}) or {}
            for lang, obj in desc.items():
                if isinstance(obj, dict) and obj.get("value"):
                    descriptions[lang] = obj.get("value")
    return {"search_top": top, "sitelinks": sitelinks, "descriptions": descriptions}

def reduce_tavily(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    res = payload.get("results") or []
    out = []
    for r in res[:5]:
        out.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "content": r.get("content")
        })
    return out

def reduce_osm(payload: Any) -> List[Dict[str, Any]]:
    arr = payload if isinstance(payload, list) else []
    out = []
    for x in arr[:3]:
        out.append({
            "display_name": x.get("display_name"),
            "class": x.get("class"),
            "type": x.get("type"),
            "lat": x.get("lat"),
            "lon": x.get("lon"),
            "address": x.get("address")
        })
    return out

# ============ LLM: relevance & merge/extract ============

def llm_relevance_gate(row_idx: int, input_row: MDMData, queries: QueryData, compact_tools: Dict[str, Any]) -> Dict[str, Any]:
    sys = (
        "You are a precise evaluator. Return STRICT JSON only.\n"
        "Schema:\n"
        '{ "relevant": true|false, "why": "short reason", "picks": { "kg_index": null|0|1|2, '
        '"wikidata_use_top": true|false, "tavily_indexes": [ints], "osm_index": null|0|1|2 } }\n'
        "If unsure, set relevant=false and explain in 'why'."
    )
    usr = {
        "task": "Determine if tool results plausibly refer to the same organization as the input.",
        "input_company": asdict(input_row),
        "queries": asdict(queries),
        "tools_summary": compact_tools
    }

    if MERGE_DEBUG:
        debug_dump_raw(row_idx, "gate_request", usr, ext="json")
        print(f'    MERGE|gate input sizes: kg={len(compact_tools.get("google_kg") or [])}, '
              f'wd={1 if compact_tools.get("wikidata") else 0}, '
              f'tv={len(compact_tools.get("tavily") or [])}, '
              f'osm={len(compact_tools.get("osm") or [])}')

    try:
        t0 = time.time()
        content = openai_chat(
            [{"role": "system", "content": sys},
             {"role": "user", "content": json.dumps(usr, ensure_ascii=False)}],
            model="gpt-4o-mini",
            temperature=0.0,
            json_mode=True,
        )
        dt_ms = int((time.time() - t0) * 1000)
        if MERGE_DEBUG:
            debug_dump_raw(row_idx, "gate_response", {"raw": content}, ext="json")
        data = json.loads(content)
        if MERGE_DEBUG:
            print(f"    MERGE|gate picks: relevant={data.get('relevant')} in {dt_ms}ms | picks={data.get('picks')}")
        return data
    except Exception as e:
        if MERGE_DEBUG:
            print(f"    MERGE|gate json_error: {e}")
        return {"relevant": False, "why": "llm_json_error", "picks": {}}

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
        "confidence": 0.0
    }

def _coerce_output_mdm(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    base = _blank_output_mdm()
    missing = []
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
        mdm_keys = {"canonical_name", "address", "city", "state", "postal_code", "country"}
        if any(k in data for k in mdm_keys):
            return data, [], "", {}
    return {}, [], "", {}

def _stringish(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())

def _flatten_tool_texts(compact_tools: Dict[str, Any]) -> str:
    """Used only for optional sanity checks."""
    chunks: List[str] = []
    for k, v in (compact_tools or {}).items():
        try:
            chunks.append(json.dumps(v, ensure_ascii=False))
        except Exception:
            pass
    big = "\n".join(chunks)
    return big

def _drop_unprovenanced_fields(out: Dict[str, Any], provenance: Any) -> Tuple[Dict[str, Any], List[str]]:
    """
    Drop any field that lacks provenance entries.
    Expected provenance structure (flexible, from LLM):
      {
        "canonical_name": [ { "source": "KG|Wikidata|Tavily|OSM", "text": "...", "url": "..." }, ... ],
        "address": [ ... ],
        ...
      }
    If a field has an empty list or missing key -> drop (set to blank/None).
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

    # lat/lon (require provenance too)
    if result.get("lat") is not None and not _has_prov("lat"):
        result["lat"] = None
        dropped.append("lat")
    if result.get("lon") is not None and not _has_prov("lon"):
        result["lon"] = None
        dropped.append("lon")

    # confidence left as-is
    return result, dropped

def llm_merge_extract(row_idx: int, input_row: MDMData, queries: QueryData, compact_tools: Dict[str, Any], picks: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str, bool, Any]:
    """
    Returns: (output_mdm, evidence, notes, ok_llm, provenance)
    NO deterministic fallback.
    """
    sys = (
        "You are a careful data assembler. Return STRICT JSON only (json_object).\n"
        "RULES (very important):\n"
        "1) Only include values that are PRESENT in the tool snippets (Google KG, Wikidata, Tavily results, OSM).\n"
        "2) Do NOT infer or guess. If a field is not explicitly supported by evidence, leave it blank/empty.\n"
        "3) Do NOT copy values from the INPUT unless the exact (or trivially normalized) string also appears in at least one tool snippet.\n"
        "4) Prefer authoritative sources. If conflicting, leave blank or choose the one with the clearest citation.\n"
        "5) Provide per-field 'provenance' listing the source items and quotes/URLs you used.\n"
        "6) If nothing reliable is found, return empty fields.\n"
        "RESPONSE FORMAT (preferred):\n"
        '{\n'
        '  \"output_mdm\": {\"canonical_name\":\"\", \"aka\":[], \"address\":\"\", \"city\":\"\", \"state\":\"\", \"postal_code\":\"\", \"country\":\"\", \"lat\":null, \"lon\":null, \"websites\":[], \"ids\":{}, \"confidence\":0.0},\n'
        '  \"evidence\": [ {\"source\":\"...\",\"url\":\"...\",\"note\":\"...\"} ],\n'
        '  \"provenance\": { \"field_name\": [ {\"source\":\"KG|Wikidata|Tavily|OSM\",\"text\":\"...\",\"url\":\"...\"} ] },\n'
        '  \"notes\": \"short rationale\"\n'
        '}\n'
        "Returning the OutputMDM object directly is allowed, but provenance is strongly preferred."
    )
    usr = {
        "instruction": "Assemble OutputMDM using ONLY evidence from tools. No inference. If uncertain, leave blank.",
        "input_company": asdict(input_row),  # for reference only
        "queries": asdict(queries),
        "tools_summary": compact_tools,
        "picks": picks,
        "expected_schema": _blank_output_mdm()
    }

    if MERGE_DEBUG:
        summary = {
            "input_company_keys": list(asdict(input_row).keys()),
            "queries": asdict(queries),
            "sizes": {
                "kg": len(compact_tools.get("google_kg") or []),
                "tavily": len(compact_tools.get("tavily") or []),
                "osm": len(compact_tools.get("osm") or []),
                "wd_has": bool(compact_tools.get("wikidata"))
            },
            "picks": picks
        }
        debug_dump_raw(row_idx, "merge_request_summary", summary, ext="json")
        print(f'    MERGE|request summary: kg={summary["sizes"]["kg"]}, '
              f'wd={int(summary["sizes"]["wd_has"])}, tv={summary["sizes"]["tavily"]}, '
              f'osm={summary["sizes"]["osm"]}')

    raw_text = ""
    try:
        t0 = time.time()
        raw_text = openai_chat(
            [{"role": "system", "content": sys},
             {"role": "user", "content": json.dumps(usr, ensure_ascii=False)}],
            model="gpt-4o-mini",
            temperature=0.0,
            json_mode=True,
        )
        elapsed = int((time.time() - t0) * 1000)

        if MERGE_DEBUG:
            debug_dump_raw(row_idx, "merge_raw", raw_text, ext="txt")
            head = raw_text[:600].replace("\n", " ")
            print(f"    MERGE|raw response len={len(raw_text)} chars ({elapsed}ms) | head600='{head}'")

        data = json.loads(raw_text)
        out_raw, ev, notes, provenance = _unwrap_merge_response(data)

        coerced, missing_keys = _coerce_output_mdm(out_raw)
        filled_keys = [k for k, v in coerced.items() if k not in {"aka","websites","ids","confidence"} and bool(v)]
        if MERGE_DEBUG:
            print(f"    MERGE|json parse: success | missing_keys={missing_keys}")
            print(f"    MERGE|filled keys: {filled_keys} | websites={len(coerced.get('websites') or [])} | ids={list((coerced.get('ids') or {}).keys())} | conf={coerced.get('confidence')}")

        if MERGE_DEBUG or DEBUG_RAW:
            debug_dump_raw(row_idx, "merge_provenance", provenance or {}, ext="json")

        return coerced, (ev if isinstance(ev, list) else []), (notes or ""), True, provenance

    except Exception as e:
        if MERGE_DEBUG:
            print(f"    MERGE|json parse/error: {e} | raw_len={len(raw_text)}")
        return _blank_output_mdm(), [], "merge_llm_json_error", False, {}

# ============ Queries (derived only; input untouched) ============

def build_queries(m: MDMData) -> QueryData:
    cty, cty_city = treat_hong_kong(m.country, m.city)
    country_for_q = country_label(cty)
    q_name = normalize_text(m.name)
    q_name_geo = " ".join([x for x in [q_name, normalize_text(cty_city), country_for_q] if x]).strip()
    q_full_addr = build_full_addr(normalize_text(m.address), normalize_text(m.city),
                                  normalize_text(m.state), normalize_text(m.postal_code),
                                  country_for_q)
    return QueryData(q_name=q_name, q_name_geo=q_name_geo, q_full_addr=q_full_addr)

# ============ IO ============

def read_input_rows(path: str) -> List[MDMData]:
    df = pd.read_excel(path)
    rows: List[MDMData] = []
    for _, r in df.iterrows():
        rows.append(MDMData(
            name=str(r.get("SOURCE_NAME", "") or ""),
            address=str(r.get("SOURCE_ADDRESS", "") or ""),
            city=str(r.get("SOURCE_CITY", "") or ""),
            state=str(r.get("SOURCE_STATE", "") or ""),
            country=str(r.get("SOURCE_COUNTRY", "") or ""),
            postal_code=str(r.get("SOURCE_POSTAL_CODE", "") or "")
        ))
    return rows

# ============ Row processing ============

def process_row(idx: int, m: MDMData) -> OutputRow:
    t0 = time.time()
    q = build_queries(m)

    # ---- Tools ----
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

    opencorporates_stub(m.name, m.country)  # stub

    compact_tools = {
        "google_kg": kg_reduced,
        "wikidata": wd_reduced,
        "tavily": tav_reduced,
        "osm": osm_reduced
    }

    # ---- LLM Relevance Gate ----
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
            enriched_mdm={}
        )

    # ---- LLM Merge/Extract (no fallback) ----
    picks = rel.get("picks", {}) if isinstance(rel.get("picks"), dict) else {}
    output_mdm, evidence, notes_merge, ok_llm, provenance = llm_merge_extract(idx, m, q, compact_tools, picks)

    enriched = {}
    websites: List[str] = []
    evidence_count = 0
    notes_final = notes_merge or ""

    if ok_llm and not _is_effectively_empty_output(output_mdm):
        # Optional strict enforcement: drop fields with no provenance
        dropped = []
        if PROVENANCE_ENFORCE:
            output_mdm, dropped = _drop_unprovenanced_fields(output_mdm, provenance)
            if dropped:
                msg = f"dropped_unverified: {', '.join(dropped)}"
                notes_final = (notes_final + ("; " if notes_final else "") + msg).strip()
        if not _is_effectively_empty_output(output_mdm):
            enriched = output_mdm
            websites = output_mdm.get("websites") or []
            # evidence_count: prefer provenance counts if present
            if isinstance(provenance, dict) and provenance:
                evidence_count = sum(len(v) for v in provenance.values() if isinstance(v, list))
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
        enriched_mdm=enriched
    )

def _is_effectively_empty_output(out: Dict[str, Any]) -> bool:
    if not isinstance(out, dict) or not out:
        return True
    keys = ["canonical_name", "address", "city", "state", "postal_code", "country"]
    if any(out.get(k) for k in keys):
        return False
    if out.get("websites") or out.get("ids"):
        return False
    return True

# ============ Batch API (for driver.py) ============

def run_batch(rows: List[MDMData], minimal_logging: bool = True) -> List[Dict[str, Any]]:
    """
    Library entrypoint for driver.py:
      - Processes ALL provided rows (no internal cap)
      - Returns a list of dicts (asdict(OutputRow)) EXACTLY as produced per row
      - Does NOT write files
    """
    results: List[Dict[str, Any]] = []
    for idx, m in enumerate(rows):
        if minimal_logging:
            print(f'[ROW {idx}] "{(m.name or "").strip()}" ({(m.city or "").strip()}, {(m.country or "").strip()})')
        try:
            out_row = process_row(idx, m)
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
                enriched_mdm={}
            )
            print(f"  !! Error processing row {idx}: {e}")
        results.append(asdict(out_row))
    return results
