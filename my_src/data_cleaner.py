from pathlib import Path
from typing import Dict,List,Optional
import pandas as pd

class DataCleaner:
    def __init__(self,df: pd.DataFrame,name: str="dataset"):
        if not isinstance(df,pd.DataFrame):
            raise TypeError('df mystr be a pandas DataFrame')

        self.original_df=df.copy()
        self.df=df.copy()
        self.name=name
        self.cleaning_log=[]

    def _log(self,step: str,details: Dict):
        self.cleaning_log.append({
            "step": step,
            "details": details
        })

    def standardize_column_names(self):
        old_columns = self.df.columns.tolist()

        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

        self._log("standardize_column_names", {
            "old_columns": old_columns,
            "new_columns": self.df.columns.tolist()
        })

        return self

    def remove_exact_duplicates(self):
        before = len(self.df)

        self.df = self.df.drop_duplicates()

        after = len(self.df)

        self._log("remove_exact_duplicates", {
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after
        })
        return self

    def remove_duplicates_by_key(self, subset: List[str], keep: str = "first"):
        before = len(self.df)

        self.df = self.df.drop_duplicates(
            subset=subset,
            keep=keep
        )

        after = len(self.df)

        self._log("remove_duplicates_by_key", {
            "subset": subset,
            "keep": keep,
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after
        })

        return self

    def drop_missing(self, columns: Optional[List[str]] = None):
        before = len(self.df)

        self.df = self.df.dropna(subset=columns)

        after = len(self.df)

        self._log("drop_missing", {
            "columns": columns,
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after
        })

        return self

    def fill_missing_values(self, columns, method="constant", value=None):
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"{col} not found in dataframe")

            if method == "constant":
                self.df[col] = self.df[col].fillna(value)

            elif method == "mean":
                self.df[col] = self.df[col].fillna(self.df[col].mean())

            elif method == "median":
                self.df[col] = self.df[col].fillna(self.df[col].median())

            elif method == "mode":
                mode_value = self.df[col].mode(dropna=True)

                if not mode_value.empty:
                    self.df[col] = self.df[col].fillna(mode_value[0])

            elif method == "ffill":
                self.df[col] = self.df[col].ffill()

            elif method == "bfill":
                self.df[col] = self.df[col].bfill()

            elif method == "interpolate":
                self.df[col] = self.df[col].interpolate()

            else:
                raise ValueError(
                    "method must be one of: constant, mean, median, mode, ffill, bfill, interpolate"
                )

        self._log("fill_missing_values", {
            "columns": columns,
            "method": method,
            "value": value
        })

        return self

    def strip_text_columns(self,columns):
        object_cols = columns

        for col in object_cols:
            self.df[col] = self.df[col].astype("string").str.strip()

        self._log("strip_text_columns", {
            "columns": object_cols
        })

        return self

    def lowercase_text_columns(self, columns: List[str]):
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.lower()

        self._log("lowercase_text_columns", {
            "columns": columns
        })

        return self



    def convert_column_types(self, type_mapping: dict):

        conversion_log = {}

        for column, target_type in type_mapping.items():

            target_type = target_type.lower()

            if column not in self.df.columns:
                raise ValueError(
                    f"{column} not found in dataframe"
                )

            old_type = str(self.df[column].dtype)

            if target_type == "integer":

                self.df[column] = (
                    pd.to_numeric(
                        self.df[column],
                        errors="coerce"
                    )
                    .astype("Int64")
                )


            elif target_type == "float":

                self.df[column] = pd.to_numeric(
                    self.df[column],
                    errors="coerce"
                )


            elif target_type == "string":

                self.df[column] = (
                    self.df[column]
                    .astype("string")
                )


            elif target_type == "object":

                self.df[column] = (
                    self.df[column]
                    .astype("object")
                )


            elif target_type == "category":

                self.df[column] = (
                    self.df[column]
                    .astype("category")
                )


            elif target_type == "datetime":

                self.df[column] = pd.to_datetime(
                    self.df[column],
                    errors="coerce"
                )


            elif target_type == "boolean":

                self.df[column] = (
                    self.df[column]
                    .astype("boolean")
                )

            elif target_type == "datetime":

                self.df[column] = pd.to_datetime(
                    self.df[column],
                    errors="coerce"
                )
            else:
                raise ValueError(
                    f"{target_type} is not supported"
                )

            conversion_log[column] = {
                "from": old_type,
                "to": str(self.df[column].dtype)
            }

        self._log(
            "convert_column_types",
            conversion_log
        )

        return self

    def standardize_categories(self, column: str, mapping: dict):
        if column not in self.df.columns:
            raise ValueError(f"{column} not found in dataframe")

        before_unique = self.df[column].nunique(dropna=False)

        self.df[column] = (
            self.df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace(mapping)

        )

        after_unique = self.df[column].nunique(dropna=False)

        self._log("standardize_categories", {
            "column": column,
            "mapping": mapping,
            "unique_before": before_unique,
            "unique_after": after_unique
        })

        return self

    def get_clean_data(self):
        return self.df.copy()

    def get_cleaning_log(self):
        return pd.DataFrame(self.cleaning_log)

    def save_clean_data(self, output_folder):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        file_path = output_folder / f"{self.name}_clean.csv"

        self.df.to_csv(file_path, index=False)

        print(f"Saved clean data -> {file_path}")

        return file_path

