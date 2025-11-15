import pandas as pd

class GenerateCSVTool:
    """Writes input_data and search_result_data side-by-side with prefixed column names."""

    def _to_df(self, data):
        if isinstance(data, dict):
            return pd.DataFrame([data])
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise TypeError("data must be a dict or list of dicts")

    def _sanitize_col(self, col):
        return str(col).strip().replace(" ", "_")

    def run(
        self,
        input_data,
        search_result_data,
        out_path="output.csv",
        input_prefix="input_",
        output_prefix="output_",
    ):
        """Create CSV with prefixed input/output columns side-by-side.

        Args:
            input_data: dict or list[dict] for the left-hand (input) side.
            search_result_data: dict or list[dict] for the right-hand (search result) side.
            out_path: output CSV path.
        Returns:
            The path to the written CSV file.
        """
        df_input = self._to_df(input_data).reset_index(drop=True)
        df_search = self._to_df(search_result_data).reset_index(drop=True)

        len_input = len(df_input)
        len_search = len(df_search)
        max_len = max(len_input, len_search, 1)

        # replicate single-row side when the other side is longer
        if len_input == 1 and len_search > 1:
            df_input = pd.DataFrame([df_input.iloc[0].to_dict()] * len_search)
        elif len_search == 1 and len_input > 1:
            df_search = pd.DataFrame([df_search.iloc[0].to_dict()] * len_input)
        else:
            if len_input != max_len:
                df_input = df_input.reindex(range(max_len)).reset_index(drop=True)
            if len_search != max_len:
                df_search = df_search.reindex(range(max_len)).reset_index(drop=True)

        # Build rename maps with sanitized column names
        input_rename = {col: f"{input_prefix}{self._sanitize_col(col)}" for col in df_input.columns}
        output_rename = {col: f"{output_prefix}{self._sanitize_col(col)}" for col in df_search.columns}

        df_input_renamed = df_input.rename(columns=input_rename)
        df_search_renamed = df_search.rename(columns=output_rename)

        # Concatenate side-by-side with input columns first
        out_df = pd.concat([df_input_renamed.reset_index(drop=True), df_search_renamed.reset_index(drop=True)], axis=1)

        out_df.to_csv(out_path, index=False)
        return out_path


if __name__ == "__main__":
    csv_tool = GenerateCSVTool()

    sample_input = [
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
        }
    ]

    sample_search_results = [
        {
            "parent company": "Amazon",
            "company address": "33 Hellow World Ln",
            "company state": "YT",
            "company country": "US",
            "postal code": "23245"
        },
        {
            "parent company": "Amazon",
            "company address": "44 Another Ave",
            "company state": "NC",
            "company country": "US",
            "postal code": "28211"
        }
    ]

    out_file = csv_tool.run(sample_input, sample_search_results, out_path="example_prefixed.csv")
    print("CSV saved to:", out_file)
