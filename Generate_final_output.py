from dotenv import load_dotenv
from groq import Groq
import os
from __future__ import annotations
import json
from dotenv import load_dotenv

# load secrets
load_dotenv()


class GenerateFinalOutputTool:
    """Tool to load a comparison JSON and ask Groq to summarize the result.

    Usage:
        tool = GenerateFinalOutputTool(groq_api_key=os.getenv('GROQ_API_KEY'))
        summary = tool.run('comparison_result.json')
    """

    def __init__(self, groq_api_key: str):
        if not groq_api_key:
            raise ValueError("groq_api_key is required")
        self.groq_api_key = groq_api_key

    def _load_client(self) -> Groq:
        try:
            return Groq(api_key=self.groq_api_key)
        except Exception as e:
            raise ImportError("Failed to create Groq client: {}".format(e))

    def run(self, json_path: str) -> str:
        """Load the JSON comparison at json_path and produce a natural-language summary.

        The JSON is expected to contain at least a top-level `summary` and optional `comparison`.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Comparison file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data.get("summary", {})
        comparison = data.get("comparison", [])

        # Build a prompt that directly uses the extracted `summary` and a
        # small preview of `comparison` so those variables are used and the
        # prompt is clearer for the LLM.
        summary_str = json.dumps(summary, indent=2)
        # Show up to the first 5 comparison entries as a preview
        try:
            comparison_preview = json.dumps(comparison[:5], indent=2)
        except Exception:
            comparison_preview = json.dumps(comparison, indent=2)

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

Comparison preview (first up to 5 items):
{comparison_preview}

Full JSON (for reference):
{json.dumps(data, indent=2)}
"""

        client = self._load_client()

        # Use the Groq chat completions API to get a textual summary.
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You summarize company data comparisons."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        # Response parsing: support common Groq shapes
        try:
            return response.choices[0].message["content"]
        except Exception:
            # Fallback: str(response)
            return str(response)


if __name__ == "__main__":
    key = os.getenv("GROQ_API_KEY")
    tool = GenerateFinalOutputTool(groq_api_key=key)
    try:
        summary = tool.run("comparison_result.json")
        print(summary)
    except Exception as e:
        print("Error:", e)