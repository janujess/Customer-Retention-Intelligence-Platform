from pathlib import Path
from typing import Dict
import pandas as pd

class DataProfiler:
    def __init__(self, df: pd.DataFrame,name:str="dataset"):
        self.df=df.copy()
        self.name=name

    def quick_peek(self):
        return self.df.head(5)


    def shape_summary(self):
        return pd.DataFrame([{
            "dataset":self.name,
            "rows":self.df.shape[0],
            "columns": self.df.shape[1]
        }])

    def column_summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "column": self.df.columns,
            "dtype": self.df.dtypes.astype(str).values,
            "missing_count": self.df.isna().sum().values,
            "missing_percent": (self.df.isna().mean() * 100).round(2).values,
            "unique_count": self.df.nunique(dropna=True).values,
            "unique_percent": (self.df.nunique(dropna=True) / len(self.df) * 100).round(2).values
        })

    def duplicate_summary(self,subset=None) -> pd.DataFrame:
        duplicate_count = self.df.duplicated().sum()

        return pd.DataFrame([{
            "dataset": self.name,
            "duplicate_rows": int(duplicate_count),
            "duplicate_percent": round(duplicate_count / len(self.df) * 100, 2)
        }])

    def numeric_summary(self) -> pd.DataFrame:
        numeric_df = self.df.select_dtypes(include="number")

        if numeric_df.empty:
            return pd.DataFrame()

        return numeric_df.describe().T

    def categorical_summary(self) ->pd.DataFrame:
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns

        summaries = {}

        for col in categorical_cols:
            summaries[col] = (
                self.df[col]
                .value_counts(dropna=False)
                .head(10)
                .reset_index()
            )

            summaries[col].columns = [col, "count"]

        return pd.DataFrame()

    def date_column_candidates(self):
        return [
            col for col in self.df.columns
            if "date" in col.lower() or "time" in col.lower()
        ]

    def outlier_summary_iqr(self) -> pd.DataFrame:
        numeric_df = self.df.select_dtypes(include="number")

        results = []

        for col in numeric_df.columns:
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = numeric_df[
                (numeric_df[col] < lower) |
                (numeric_df[col] > upper)
                ][col]

            results.append({
                "column": col,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": outliers.count(),
                "outlier_percent": round(outliers.count() / len(self.df) * 100, 2)
            })

        return pd.DataFrame(results)

    def cardinality_summary(
            self,
            columns=None,
            top_n=10,
            max_cardinality=20
    ):

        if columns is None:
            columns = self.df.select_dtypes(
                include=[
                    "object",
                    "string",
                    "category",
                    "bool"
                ]
            ).columns.tolist()

        cardinality_results = []
        top_values_tables = {}

        for col in columns:

            if col not in self.df.columns:
                raise ValueError(
                    f"{col} not found in dataframe"
                )

            unique_count = self.df[col].nunique(
                dropna=True
            )

            unique_percent = round(
                unique_count / len(self.df) * 100,
                2
            )

            cardinality_results.append(
                {
                    "column": col,
                    "unique_count": unique_count,
                    "unique_percent": unique_percent
                }
            )

            # only inspect true categorical columns
            if unique_count <= max_cardinality:
                values = (
                    self.df[col]
                    .value_counts(dropna=False)
                    .head(top_n)
                    .reset_index()
                )

                values.columns = [
                    col,
                    "count"
                ]

                values["percentage"] = (
                        values["count"]
                        / len(self.df)
                        * 100
                ).round(2)

                top_values_tables[col] = values

        cardinality_df = pd.DataFrame(
            cardinality_results
        )

        return (
            cardinality_df,
            top_values_tables
        )

    def imbalance_summary(self, columns=None):

        if columns is None:
            columns = self.df.select_dtypes(
                include=[
                    "object",
                    "string",
                    "category",
                    "bool"
                ]
            ).columns.tolist()

        results = []

        for col in columns:
            counts = self.df[col].value_counts(
                dropna=False
            )

            percentages = (
                    self.df[col]
                    .value_counts(
                        normalize=True,
                        dropna=False
                    )
                    * 100
            )

            results.append({

                "column": col,

                "majority_class": counts.index[0],

                "majority_percent": round(
                    percentages.iloc[0],
                    2
                ),

                "minority_class": counts.index[-1],

                "minority_percent": round(
                    percentages.iloc[-1],
                    2
                ),

                "imbalance_ratio": round(
                    counts.iloc[0]
                    /
                    counts.iloc[-1],
                    2
                )

            })

        return pd.DataFrame(results)

    def full_profile(self) -> Dict:
        return {
            "shape": self.shape_summary(),
            "columns": self.column_summary(),
            "duplicates": self.duplicate_summary(),
            "numeric": self.numeric_summary(),
            "categoricals": self.categorical_summary(),
            "date_candidates": self.date_column_candidates(),
            "outliers": self.outlier_summary_iqr()
        }

    def save_profile(self, output_folder):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        self.column_summary().to_csv(
            output_folder / f"{self.name}_column_summary.csv",
            index=False
        )

        self.numeric_summary().to_csv(
            output_folder / f"{self.name}_numeric_summary.csv"
        )

        self.outlier_summary_iqr().to_csv(
            output_folder / f"{self.name}_outlier_summary.csv",
            index=False
        )

        print(f"Saved profile for {self.name}")