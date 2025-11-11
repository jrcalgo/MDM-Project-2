import pandas as pd
import os
from __future__ import annotations
import json
import pandas as pd
from typing import Dict, Any


# ----------------------------------------------------
# Generate CSV Tool
# ----------------------------------------------------
class GenerateCSVTool:
    """Saves a single record or list of records to CSV."""


    def run(self, data: Dict[str, Any] | list[Dict[str, Any]], out_path: str = "output.csv") -> str:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        df.to_csv(out_path, index=False)

        return out_path

# ----------------------------------------------------
# Example usage (not executed automatically)
# ----------------------------------------------------
if __name__ == "__main__":
    csv_tool = GenerateCSVTool()
    csv_tool.run(
    data={
        "company": "WXY",
        "parent": "Amazon",
        "state": "NY",
        "country": "USA",
        "postal": "10001",
    },
    out_path="example.csv"
    )