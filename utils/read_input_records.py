from typing import List, Dict, Any, Union, Tuple
import pandas as pd

from data_classes.mdm import MDMData


def read_input_rows(
    path: str,
    use_dnb: bool = False,
) -> Union[List[MDMData], Tuple[List[MDMData], List[Dict[str, Any]]]]:
    """
    Read the source input file and construct MDMData records.

    When use_dnb=False (default):
        -> returns List[MDMData]  (backwards compatible)

    When use_dnb=True:
        -> returns (List[MDMData], List[Dict[str, Any]])
           where the second list is the per-row DNB/PRIMARY payload.
    """

    path_lower = path.lower()
    if path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        # CSV default
        df = pd.read_csv(path, dtype=str, encoding="utf-8", keep_default_na=False)

    df = df.fillna("")

    records: List[MDMData] = []
    dnb_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        r = row.to_dict()

        # === SOURCE → MDMData (same mapping as your original version) ===
        mdm_record = MDMData(
            name=str(r.get("SOURCE_NAME", "") or ""),
            address=str(r.get("SOURCE_ADDRESS", "") or ""),
            city=str(r.get("SOURCE_CITY", "") or ""),
            state=str(r.get("SOURCE_STATE", "") or ""),
            country=str(r.get("SOURCE_COUNTRY", "") or ""),
            postal_code=str(r.get("SOURCE_POSTAL_CODE", "") or ""),
        )
        records.append(mdm_record)

        # === NEW: build DNB/PRIMARY dict when flag is on ===
        if use_dnb:
            dnb_payload: Dict[str, Any] = {
                # You’ll adjust these column names to match query_group2.csv
                "primary_name": r.get("ORGANIZATIONPRIMARYNAME", ""),
                "primary_address": r.get("PRIMARYSTREETADDRESSLINE1", ""),
                "primary_city": r.get("PRIMARYTOWNNAME", ""),
                "primary_state": r.get("PRIMARYADDRESSREGIONNAME", ""),
                "primary_postal_code": r.get("PRIMARYPOSTALCODE", ""),
                "primary_country": r.get("PRIMARYCOUNTRYISOALPHA2CD", ""),
                # add any other PRIMARY_* / DNB_* fields you want here
            }
            dnb_rows.append(dnb_payload)

    if use_dnb:
        return records, dnb_rows
    return records
