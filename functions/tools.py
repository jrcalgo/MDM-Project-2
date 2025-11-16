# functions/tools.py

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import requests

# Separate USER_AGENT here so we don't create circular imports
USER_AGENT = os.getenv("MDM_USER_AGENT", "MDM-Enricher/1.0")


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
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": name,
            "limit": 3,
        }
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params=params,
            timeout=45,
            headers={"User-Agent": USER_AGENT},
        )
        return r.status_code, (r.json() if r.content else {})
    except Exception as e:
        return (-1, {"error": str(e)})


def wikidata_entity(qid: str) -> Tuple[int, Dict[str, Any]]:
    try:
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": qid,
            "props": "sitelinks|labels|aliases|descriptions|claims",
        }
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params=params,
            timeout=45,
            headers={"User-Agent": USER_AGENT},
        )
        return r.status_code, (r.json() if r.content else {})
    except Exception as e:
        return (-1, {"error": str(e)})


def google_kg_search(query: str) -> Tuple[int, Dict[str, Any]]:
    api_key = os.getenv("GOOGLE_KG_API_KEY", "")
    if not api_key:
        return (0, {"error": "no_api_key"})
    try:
        params = {"query": query, "limit": 3, "indent": True, "key": api_key}
        r = requests.get(
            "https://kgsearch.googleapis.com/v1/entities:search",
            params=params,
            timeout=45,
            headers={"User-Agent": USER_AGENT},
        )
        return r.status_code, (r.json() if r.content else {})
    except Exception as e:
        return (-1, {"error": str(e)})


def osm_geocode(addr: str) -> Tuple[int, Any]:
    try:
        params = {"q": addr, "format": "json", "limit": 3, "addressdetails": 1}
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            timeout=45,
            headers={"User-Agent": USER_AGENT},
        )
        return r.status_code, (r.json() if r.content else [])
    except Exception as e:
        return (-1, {"error": str(e)})


def opencorporates_stub(name: str, country: str) -> Tuple[int, Dict[str, Any]]:
    # Placeholder for future real OpenCorporates integration
    return (0, {"stub": True})


# ============ Reducers for LLM prompts ============

def reduce_kg(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = payload.get("itemListElement", []) or []
    out: List[Dict[str, Any]] = []
    for it in items[:3]:
        res = it.get("result", {}) or {}
        out.append(
            {
                "name": res.get("name"),
                "type": res.get("@type"),
                "description": (res.get("description") or ""),
                "detailed_url": (res.get("detailedDescription", {}) or {}).get("url"),
                "kg_id": res.get("@id"),
            }
        )
    return out


def reduce_wikidata(
    search_payload: Dict[str, Any],
    entity_payload: Dict[str, Any],
) -> Dict[str, Any]:
    search = search_payload.get("search", []) or []
    top = None
    if search:
        top = {
            "id": search[0].get("id"),
            "label": search[0].get("label"),
            "description": search[0].get("description"),
        }

    sitelinks: Dict[str, Any] = {}
    descriptions: Dict[str, Any] = {}

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
    out: List[Dict[str, Any]] = []
    for r in res[:5]:
        out.append(
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
            }
        )
    return out


def reduce_osm(payload: Any) -> List[Dict[str, Any]]:
    arr = payload if isinstance(payload, list) else []
    out: List[Dict[str, Any]] = []
    for x in arr[:3]:
        out.append(
            {
                "display_name": x.get("display_name"),
                "class": x.get("class"),
                "type": x.get("type"),
                "lat": x.get("lat"),
                "lon": x.get("lon"),
                "address": x.get("address"),
            }
        )
    return out
