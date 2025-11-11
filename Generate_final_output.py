from __future__ import annotations
from dotenv import load_dotenv
from groq import Groq
import os
import json
from typing import Any, Dict, Optional, Union

# load secrets from .env
load_dotenv()


class GenerateFinalOutputTool:
    def __init__(self, groq_api_key: str):
        if not groq_api_key:
            raise ValueError("groq_api_key is required")
        self.groq_api_key = groq_api_key

    def _load_client(self) -> Groq:
        try:
            return Groq(api_key=self.groq_api_key)
        except Exception as e:
            raise ImportError(f"Failed to create Groq client: {e}") from e

    def _load_data(self, json_input: Union[str, os.PathLike, Dict[str, Any], Any]) -> Dict[str, Any]:
        # File path
        if isinstance(json_input, (str, os.PathLike)):
            path = str(json_input)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Comparison file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Dict
        if isinstance(json_input, dict):
            return json_input

        # File-like object
        if hasattr(json_input, "read"):
            try:
                return json.load(json_input)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON from file-like object: {e}")

        raise TypeError("json_input must be a path, a dict, or a file-like object")

    def run(self, json_input: Union[str, os.PathLike, Dict[str, Any], Any]) -> str:
        data = self._load_data(json_input)

        summary = data.get("summary", {})
        comparison = data.get("comparison", [])

        if not isinstance(comparison, list):
            comparison = [comparison]

        summary_str = json.dumps(summary, indent=2)
        comparison_preview = json.dumps(comparison[:5], indent=2)

        prompt = f"""
You are a data quality analyst. A JSON comparison result is provided. Summarize:
  - whether the record matched
  - how similar the data was
  - how many fields were compared
  - what fields were outdated
  - how the outdated fields could affect confidence score
  - how corrected information improves data quality

Top-level summary:
{summary_str}

Comparison preview (first 5 items):
{comparison_preview}

Full JSON (for reference):
{json.dumps(data, indent=2)}
"""

        client = self._load_client()

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You summarize company data comparisons."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        # Support multiple Groq SDK response shapes
        msg = response.choices[0].message
        return msg["content"] if isinstance(msg, dict) else msg.content


if __name__ == "__main__":
    key = os.getenv("GROQ_API_KEY")
    tool = GenerateFinalOutputTool(groq_api_key=key)

    sample_json = {
        "summary": {
            "overall_similarity": 0.87,
            "match": True,
            "fields_compared": 7
        },
        "comparison": [
            {
                "parent company": "Amazon",
                "company address": "932 California Ln",
                "company state": "CA",
                "company country": "US",
                "postal code": "12212"
            }
        ]
    }

    try:
        summary_text = tool.run(sample_json)
        print(summary_text)
    except Exception as e:
        print("Error:", e)