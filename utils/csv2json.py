import csv
import json
from typing import List, Dict, Optional

def csv_to_dicts(path):
    """
    Read a CSV file and convert each row to a dict keyed by the header names.
    Empty strings are converted to None. Whitespace around values is stripped.
    """
    rows: List[Dict[str, Optional[str]]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=",", skipinitialspace=True)
        for raw in reader:
            row = {k: (v.strip() if v is not None and v.strip() != "" else None) for k, v in raw.items()}
            rows.append(row)
    return rows


def csv_to_json(path):
    """
    Return a JSON string representing the CSV rows as a list of objects.
    """
    dicts = csv_to_dicts(path)
    return json.dumps(dicts, ensure_ascii=False, indent=2)
