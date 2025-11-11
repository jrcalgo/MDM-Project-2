from __future__ import annotations
import pandas as pd
from typing import Dict, Any

# ----------------------------------------------------
# Generate CSV Tool
# ----------------------------------------------------
class GenerateCSVTool:
    """Saves a single record or list of records to CSV."""

    def run(self, data: Dict[str, Any] | list[Dict[str, Any]], out_path: str = "output.csv") -> str:
        # Ensure valid data format
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise TypeError("data must be a dict or list of dicts")

        df.to_csv(out_path, index=False)
        return out_path


if __name__ == "__main__":
    csv_tool = GenerateCSVTool()

    sample_json = {"updated json": [
                    {
                        "parent company": "Amazon",
                        "company address": "932 California Ln",
                        "company state": "CA",
                        "company country": "US",
                        "postal code": "12212"
                    },
                    {
                        "parent company": "Amazon",
                        "company address": "111 Cal St",
                        "company state": "CA",
                        "company country": "US",
                        "postal code": "00000"
                    },
                    {
                        "parent company": "Amazon",
                        "company address": "12 Col Monro3 St",
                        "company state": "MA",
                        "company country": "US",
                        "postal code": "33441"
                    },
                    {
                        "parent company": "Amazon",
                        "company address": "33 Hellow World Ln",
                        "company state": "YT",
                        "company country": "US",
                        "postal code": "23245"
                    }
                ]
            }

    # Extract the list of records, not the wrapper dict
    out_file = csv_tool.run(sample_json["updated json"], out_path="example.csv")
    print("CSV saved to:", out_file)
