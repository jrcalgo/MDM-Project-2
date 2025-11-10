# -*- coding: utf-8 -*-
# file: searchFunctionBatchOnly.py
#
# Batch-only runner that contains the full pipeline (no imports).
# Processes ALL rows from an Excel with hard-coded headers and writes a single JSON array.
#
# Minimal logging: load/compile notices, per-row "processing" + success/fail, final summary.
#
# Env:
#   OPENAI_API_KEY        : required
#   TAVILY_API_KEY        : optional
#   GOOGLE_KG_API_KEY     : optional
#   MDM_INPUT_XLSX        : input Excel path (default: query_group2.xlsx)
#   MDM_OUT_JSON          : output JSON path (default: mdm_results.json)
#
# Expected Excel columns (exact):
#   SOURCE_NAME, SOURCE_ADDRESS, SOURCE_CITY, SOURCE_STATE, SOURCE_COUNTRY, SOURCE_POSTAL_CODE
#
# Deps:
#   pip install pandas requests langchain langchain-openai langgraph

from __future__ import annotations

import os
import re
import json
import time
import unicodedata
import traceback
from dataclasses import dataclass, asdict, field
from typing import Optional, TypedDict, Dict, Any, List

import pandas as pd
import requests

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# -------------------------
# Data models / state types
# -------------------------
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
    name_en: str
    address_en: str
    city_en: str
    state_en: str
    country_en: str
    postal_code_en: str
    queries: Dict[str, str] = field(default_factory=dict)

@dataclass
class ToolStatus:
    tool: str
    success: bool
    reason: str
    http_status: Optional[int] = None
    elapsed_ms: Optional[int] = None

@dataclass
class SourceHit:
    source: str
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    rank: Optional[int] = None

@dataclass
class OutputData:
    public_presence: bool
    websites: List[str]
    evidence: List[SourceHit]
    notes: str
    overall_success: bool
    enriched_mdm: Dict[str, Any]

class ToolOutputs(TypedDict, total=False):
    tool_status: List[ToolStatus]
    tavily: Any
    wikidata: Any
    googlekg: Any
    osm: Any
    opencorporates: Any

class GraphState(TypedDict, total=False):
    mdm: MDMData
    query_data: QueryData
    tools: ToolOutputs
    output: Optional[OutputData]

# -------------------------
# LLM & prompts
# -------------------------
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_PREPROCESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You convert possibly non-English company records into clean English JSON.\n"
     "Return ONLY valid JSON with keys: name_en,address_en,city_en,state_en,country_en,postal_code_en.\n"
     "Do not fabricate missing parts; keep strings empty if unknown."),
    ("human",
     "Record:\n{rec}\n\n"
     "Output strictly JSON with those 6 keys.")
])

_MERGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict merger. You receive:\n"
     "- English normalized record (query_data)\n"
     "- Tool statuses and raw payloads from Tavily, Wikidata, Google KG, OSM, OpenCorporates\n"
     "Your job:\n"
     "1) Identify whether there is credible evidence that the company exists online.\n"
     "2) If credible, list canonical websites and evidence items (source/url/title/snippet/rank).\n"
     "3) Produce enriched_mdm with best canonical values (only if supported by tool evidence), otherwise leave empty.\n"
     "4) Set public_presence=true if you keep any evidence; else false.\n"
     "If there isn't enough credible evidence to finalize, leave websites/evidence empty and set public_presence=false.\n"
     "Return ONLY JSON with keys: public_presence, websites, evidence, notes, overall_success, enriched_mdm.\n"
     "Notes should justify decisions briefly. NEVER invent URLs or details."),
    ("human",
     "Normalized query_data (JSON):\n{query_data}\n\n"
     "Tool status list (JSON):\n{tool_status}\n\n"
     "Tavily raw (JSON):\n{tavily}\n\n"
     "Wikidata raw (JSON):\n{wikidata}\n\n"
     "GoogleKG raw (JSON):\n{googlekg}\n\n"
     "OSM raw (JSON):\n{osm}\n\n"
     "OpenCorporates raw (JSON):\n{opencorporates}\n\n"
     "Output strictly the target JSON. If insufficient evidence, set overall_success=false.")
])

_MERGE_KEYS = ["public_presence", "websites", "evidence", "notes", "overall_success", "enriched_mdm"]

# -------------------------
# Helpers
# -------------------------
def _nfkc_ws(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _compose_full_address(addr: str, city: str, state: str, postal: str, country: str) -> str:
    parts = []
    if addr: parts.append(addr)
    line2 = " ".join([p for p in [city, state, postal] if _nfkc_ws(p)])
    if _nfkc_ws(line2): parts.append(line2)
    if _nfkc_ws(country): parts.append(country)
    return ", ".join([_nfkc_ws(p) for p in parts if _nfkc_ws(p)])

# -------------------------
# Tool wrappers
# -------------------------
_UA = {"User-Agent": "mdm-verify/0.1 (+https://example.local)"}

def _tool_try(fn, tool_name: str):
    t0 = time.time()
    try:
        data = fn()
        dt = int((time.time() - t0) * 1000)
        status = ToolStatus(tool=tool_name, success=True, reason="ok", http_status=200, elapsed_ms=dt)
        return data, status
    except requests.HTTPError as e:
        dt = int((time.time() - t0) * 1000)
        status = ToolStatus(tool=tool_name, success=False, reason="http_error", http_status=getattr(e.response, "status_code", None), elapsed_ms=dt)
        return None, status
    except Exception:
        dt = int((time.time() - t0) * 1000)
        status = ToolStatus(tool=tool_name, success=False, reason="exception", http_status=None, elapsed_ms=dt)
        return None, status

def call_tavily(query: str, limit: int = 5) -> Any:
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY")
    url = "https://api.tavily.com/search"
    payload = {"api_key": api_key, "query": query, "include_answer": False, "max_results": limit}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def call_wikidata(name: str) -> Any:
    if not _nfkc_ws(name):
        raise RuntimeError("Empty name")
    r = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={"action": "wbsearchentities", "language": "en", "format": "json", "search": name, "type": "item", "limit": 3},
        headers=_UA, timeout=20
    )
    r.raise_for_status()
    search = r.json()
    if search.get("search"):
        qid = search["search"][0]["id"]
        r2 = requests.get(
            f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
            headers=_UA, timeout=20
        )
        r2.raise_for_status()
        return {"search": search, "entity": r2.json()}
    return {"search": search, "entity": None}

def call_googlekg(query: str) -> Any:
    api_key = os.getenv("GOOGLE_KG_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_KG_API_KEY")
    r = requests.get(
        "https://kgsearch.googleapis.com/v1/entities:search",
        params={"query": query, "limit": 3, "indent": True, "key": api_key, "languages": "en", "types": "Organization"},
        headers=_UA, timeout=20
    )
    r.raise_for_status()
    return r.json()

def call_osm_forward(address: str) -> Any:
    if not _nfkc_ws(address):
        raise RuntimeError("Empty address")
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": address, "format": "json", "limit": 3, "addressdetails": 1},
        headers=_UA, timeout=20
    )
    r.raise_for_status()
    return r.json()

def call_opencorporates_stub() -> Any:
    return None

# -------------------------
# Graph nodes
# -------------------------
def _preprocess(state: GraphState) -> GraphState:
    mdm: MDMData = state["mdm"]  # type: ignore
    rec = {
        "name": mdm.name, "address": mdm.address, "city": mdm.city,
        "state": mdm.state, "country": mdm.country, "postal_code": mdm.postal_code,
    }
    chain = _PREPROCESS_PROMPT | _llm
    resp = chain.invoke({"rec": json.dumps(rec, ensure_ascii=False)})
    try:
        obj = json.loads(resp.content)
    except Exception:
        obj = {
            "name_en": _nfkc_ws(mdm.name),
            "address_en": _nfkc_ws(mdm.address),
            "city_en": _nfkc_ws(mdm.city),
            "state_en": _nfkc_ws(mdm.state),
            "country_en": _nfkc_ws(mdm.country),
            "postal_code_en": _nfkc_ws(mdm.postal_code),
        }

    qd = QueryData(
        name_en=_nfkc_ws(obj.get("name_en", "")),
        address_en=_nfkc_ws(obj.get("address_en", "")),
        city_en=_nfkc_ws(obj.get("city_en", "")),
        state_en=_nfkc_ws(obj.get("state_en", "")),
        country_en=_nfkc_ws(obj.get("country_en", "")),
        postal_code_en=_nfkc_ws(obj.get("postal_code_en", "")),
    )

    q_name = qd.name_en
    q_name_geo = _nfkc_ws(" ".join([p for p in [qd.name_en, qd.city_en, qd.country_en] if p]))
    q_full_addr = _compose_full_address(qd.address_en, qd.city_en, qd.state_en, qd.postal_code_en, qd.country_en)
    qd.queries = {"q_name": q_name, "q_name_geo": q_name_geo, "q_full_addr": q_full_addr}

    state["query_data"] = qd
    return state

def _tools(state: GraphState) -> GraphState:
    qd: QueryData = state["query_data"]  # type: ignore
    statuses: List[ToolStatus] = []
    results: ToolOutputs = {"tool_status": statuses}

    def _tavily_do():
        q = qd.queries.get("q_name_geo") or qd.queries.get("q_name") or qd.name_en
        if not _nfkc_ws(q):
            raise RuntimeError("Empty query for Tavily")
        return call_tavily(q, limit=5)
    data, st = _tool_try(_tavily_do, "tavily")
    statuses.append(st); results["tavily"] = data

    data, st = _tool_try(lambda: call_wikidata(qd.name_en), "wikidata")
    statuses.append(st); results["wikidata"] = data

    def _kg_do():
        q = qd.queries.get("q_name_geo") or qd.name_en
        return call_googlekg(q)
    data, st = _tool_try(_kg_do, "googlekg")
    statuses.append(st); results["googlekg"] = data

    data, st = _tool_try(lambda: call_osm_forward(qd.queries.get("q_full_addr", "")), "osm")
    statuses.append(st); results["osm"] = data

    data, st = _tool_try(lambda: call_opencorporates_stub(), "opencorporates")
    statuses.append(st); results["opencorporates"] = data

    state["tools"] = results
    return state

def _safe_get(d: dict, k: str, default=None):
    v = d.get(k, default)
    return v if v is not None else default

def _coerce_output(obj: dict, qd: QueryData, tools: ToolOutputs) -> OutputData:
    for k in _MERGE_KEYS:
        if k not in obj:
            obj[k] = None

    websites = _safe_get(obj, "websites", []) or []
    dedup_web: List[str] = []
    seen = set()
    for w in websites:
        w2 = (w or "").strip()
        if w2 and w2 not in seen:
            seen.add(w2); dedup_web.append(w2)

    ev_in = _safe_get(obj, "evidence", []) or []
    ev_out: List[SourceHit] = []
    if isinstance(ev_in, list):
        for i, e in enumerate(ev_in):
            try:
                ev_out.append(SourceHit(
                    source=(e.get("source") or "").strip(),
                    url=(e.get("url") or "").strip(),
                    title=(e.get("title") or None),
                    snippet=(e.get("snippet") or None),
                    rank=e.get("rank", i+1),
                ))
            except Exception:
                continue

    public_presence = bool(obj.get("public_presence")) and (len(ev_out) > 0)
    overall_success = bool(obj.get("overall_success")) and public_presence
    enriched = obj.get("enriched_mdm") or {}
    if not isinstance(enriched, dict):
        enriched = {}
    notes = obj.get("notes") or ""

    return OutputData(
        public_presence=public_presence,
        websites=dedup_web,
        evidence=ev_out,
        notes=notes,
        overall_success=overall_success,
        enriched_mdm=enriched,
    )

def _merge_llm(state: GraphState) -> GraphState:
    qd: QueryData = state["query_data"]  # type: ignore
    tools: ToolOutputs = state["tools"]  # type: ignore

    payload = {
        "query_data": json.dumps({
            "name_en": qd.name_en,
            "address_en": qd.address_en,
            "city_en": qd.city_en,
            "state_en": qd.state_en,
            "country_en": qd.country_en,
            "postal_code_en": qd.postal_code_en,
            "queries": qd.queries,
        }, ensure_ascii=False),
        "tool_status": json.dumps([asdict(ts) for ts in tools.get("tool_status", [])], ensure_ascii=False),
        "tavily": json.dumps(tools.get("tavily"), ensure_ascii=False),
        "wikidata": json.dumps(tools.get("wikidata"), ensure_ascii=False),
        "googlekg": json.dumps(tools.get("googlekg"), ensure_ascii=False),
        "osm": json.dumps(tools.get("osm"), ensure_ascii=False),
        "opencorporates": json.dumps(tools.get("opencorporates"), ensure_ascii=False),
    }

    chain = _MERGE_PROMPT | _llm
    resp = chain.invoke(payload)

    try:
        merged = json.loads(resp.content)
    except Exception:
        state["output"] = None
        return state

    out = _coerce_output(merged, qd, tools)

    if (not out.public_presence) or ((not out.websites) and (not out.evidence)):
        state["output"] = None
        return state

    state["output"] = out
    return state

# -------------------------
# Graph compile
# -------------------------
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("preprocess", _preprocess)
    g.add_node("tools", _tools)
    g.add_node("merge_llm", _merge_llm)
    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "tools")
    g.add_edge("tools", "merge_llm")
    g.add_edge("merge_llm", END)
    return g.compile()

# -------------------------
# Batch processing (minimal logging)
# -------------------------
EXPECTED_COLS = [
    "SOURCE_NAME",
    "SOURCE_ADDRESS",
    "SOURCE_CITY",
    "SOURCE_STATE",
    "SOURCE_COUNTRY",
    "SOURCE_POSTAL_CODE",
]

def _row_to_mdm(row: pd.Series) -> MDMData:
    def g(col: str) -> str:
        v = row.get(col, "")
        return "" if pd.isna(v) else str(v)
    return MDMData(
        name=g("SOURCE_NAME"),
        address=g("SOURCE_ADDRESS"),
        city=g("SOURCE_CITY"),
        state=g("SOURCE_STATE"),
        country=g("SOURCE_COUNTRY"),
        postal_code=g("SOURCE_POSTAL_CODE"),
    )

def run_batch(xlsx_path: str) -> List[Dict[str, Any]]:
    print(f"Loading: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: missing columns {missing}. Available: {list(df.columns)}")

    print("Compiling graph...")
    app = build_graph()
    total = len(df)
    print(f"Rows: {total}")

    results: List[Dict[str, Any]] = []
    for idx in range(total):
        row = df.iloc[idx].fillna("")
        mdm = _row_to_mdm(row)
        disp_name = (mdm.name or "").strip() or "(no name)"
        print(f"[{idx+1}/{total}] Processing: {disp_name} ...", end="", flush=True)

        t0 = time.time()
        try:
            out_state = app.invoke({"mdm": mdm})
            elapsed_ms = int((time.time() - t0) * 1000)

            qd = out_state.get("query_data")
            result = out_state.get("output")

            base = {
                "row_index": idx,
                "input": asdict(mdm),
                "queries": getattr(qd, "queries", {}) if qd else {},
                "timing_ms": elapsed_ms,
            }

            if result is None:
                base.update({
                    "success": False,
                    "message": "Search was not successful.",
                    "public_presence": False,
                    "websites": [],
                    "evidence_count": 0,
                    "notes": "",
                    "enriched_mdm": {},
                })
                print(" fail")
            else:
                out_dict = result.__dict__
                base.update({
                    "success": True,
                    "public_presence": bool(out_dict.get("public_presence")),
                    "websites": out_dict.get("websites", []),
                    "evidence_count": len(out_dict.get("evidence", []) or []),
                    "notes": out_dict.get("notes", ""),
                    "enriched_mdm": out_dict.get("enriched_mdm", {}),
                    "evidence": [e.__dict__ for e in (out_dict.get("evidence") or [])],
                })
                print(" ok")

            results.append(base)

        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            results.append({
                "row_index": idx,
                "input": asdict(mdm),
                "success": False,
                "message": f"Unhandled error: {e.__class__.__name__}: {e}",
                "public_presence": False,
                "websites": [],
                "evidence_count": 0,
                "notes": "",
                "enriched_mdm": {},
                "timing_ms": elapsed_ms,
            })
            print(" error")

    succ = sum(1 for r in results if r.get("success"))
    print(f"Done. total={len(results)} success={succ} failed={len(results)-succ}")
    return results

def _write_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    xlsx = (os.getenv("MDM_INPUT_XLSX") or "query_group2.xlsx").strip()
    out_json = (os.getenv("MDM_OUT_JSON") or "mdm_results.json").strip()

    all_results = run_batch(xlsx)
    _write_json(out_json, all_results)
    print(f"Output: {out_json}")
