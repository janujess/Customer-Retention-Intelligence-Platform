import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ModelPreprocessor:

    def __init__(
        self,
        target_column,
        id_column="account_id",
        date_column="snapshot_date"
    ):

        self.target_column = target_column
        self.id_column = id_column
        self.date_column = date_column

        self.preprocessor = None

        self.numeric_columns = []
        self.categorical_columns = []

        self.feature_columns = []

    def prepare_dataframe(self, df):

        data = df.copy()

        required_columns = [
            self.id_column,
            self.date_column,
            self.target_column
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )


        # Ensure snapshot date is datetime
        data[self.date_column] = pd.to_datetime(
            data[self.date_column],
            errors="raise"
        )


        # Ensure target is integer
        data[self.target_column] = (
            data[self.target_column]
            .astype(int)
        )


        # Sort chronologically
        data = (
            data
            .sort_values(
                [
                    self.date_column,
                    self.id_column
                ]
            )
            .reset_index(drop=True)
        )

        return data

    def chronological_split(
        self,
        df,
        train_size=0.70,
        validation_size=0.15,
        gap_days=7
    ):

        data = self.prepare_dataframe(df)

        if train_size <= 0 or validation_size <= 0:

            raise ValueError(
                "train_size and validation_size "
                "must both be greater than 0."
            )

        if train_size + validation_size >= 1:

            raise ValueError(
                "train_size + validation_size must be "
                "less than 1 so a test set remains."
            )

        if gap_days < 0:

            raise ValueError(
                "gap_days cannot be negative."
            )

        unique_dates = (
            data[self.date_column]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        number_of_dates = len(unique_dates)

        if number_of_dates < 3:

            raise ValueError(
                "Not enough unique snapshot dates "
                "for train/validation/test splitting."
            )

        train_end_index = int(
            number_of_dates * train_size
        )

        validation_end_index = int(
            number_of_dates
            * (train_size + validation_size)
        )


        validation_start_date = (
            unique_dates.iloc[train_end_index]
        )

        test_start_date = (
            unique_dates.iloc[validation_end_index]
        )
        train_cutoff = (
            validation_start_date
            - pd.Timedelta(days=gap_days)
        )

        validation_cutoff = (
            test_start_date
            - pd.Timedelta(days=gap_days)
        )

        train_df = (
            data[
                data[self.date_column]
                < train_cutoff
            ]
            .copy()
            .reset_index(drop=True)
        )

        validation_df = (
            data[
                (
                    data[self.date_column]
                    >= validation_start_date
                )
                &
                (
                    data[self.date_column]
                    < validation_cutoff
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        test_df = (
            data[
                data[self.date_column]
                >= test_start_date
            ]
            .copy()
            .reset_index(drop=True)
        )

        if train_df.empty:

            raise ValueError(
                "Training set is empty after applying "
                "the temporal gap."
            )

        if validation_df.empty:

            raise ValueError(
                "Validation set is empty after applying "
                "the temporal gap."
            )

        if test_df.empty:

            raise ValueError(
                "Test set is empty."
            )


        return (
            train_df,
            validation_df,
            test_df
        )

    def create_xy(self, df):

        y = (
            df[self.target_column]
            .astype(int)
            .reset_index(drop=True)
        )
        target_columns = [
            column
            for column in df.columns
            if column.startswith("churn_next_")
        ]

        columns_to_drop = [
            *target_columns,
            self.id_column,
            self.date_column,
            "signup_date"
        ]


        X = (
            df
            .drop(
                columns=columns_to_drop,
                errors="ignore"
            )
            .copy()
            .reset_index(drop=True)
        )

        datetime_columns = (
            X
            .select_dtypes(
                include=[
                    "datetime",
                    "datetimetz"
                ]
            )
            .columns
            .tolist()
        )

        if datetime_columns:

            X = X.drop(
                columns=datetime_columns
            )

        numeric_columns = (
            X
            .select_dtypes(
                include="number"
            )
            .columns
        )

        X[numeric_columns] = (
            X[numeric_columns]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )


        return X, y

    def build_preprocessor(
        self,
        X_train
    ):

        self.numeric_columns = (
            X_train
            .select_dtypes(
                include=[
                    "number",
                    "bool"
                ]
            )
            .columns
            .tolist()
        )

        self.categorical_columns = (
            X_train
            .select_dtypes(
                include=[
                    "object",
                    "category",
                    "string"
                ]
            )
            .columns
            .tolist()
        )

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_columns
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_columns
                )
            ],
            remainder="drop"
        )


        return self.preprocessor

    def create_metadata(
        self,
        df
    ):

        return (
            df[
                [
                    self.id_column,
                    self.date_column,
                    self.target_column
                ]
            ]
            .copy()
            .reset_index(drop=True)
        )

    def prepare_model_data(
        self,
        df,
        train_size=0.70,
        validation_size=0.15,
        gap_days=7
    ):

        (
            train_df,
            validation_df,
            test_df
        ) = self.chronological_split(
            df=df,
            train_size=train_size,
            validation_size=validation_size,
            gap_days=gap_days
        )

        train_metadata = (
            self.create_metadata(
                train_df
            )
        )

        validation_metadata = (
            self.create_metadata(
                validation_df
            )
        )

        test_metadata = (
            self.create_metadata(
                test_df
            )
        )

        X_train, y_train = (
            self.create_xy(
                train_df
            )
        )

        X_validation, y_validation = (
            self.create_xy(
                validation_df
            )
        )

        X_test, y_test = (
            self.create_xy(
                test_df
            )
        )

        self.feature_columns = (
            X_train.columns.tolist()
        )

        X_validation = (
            X_validation.reindex(
                columns=self.feature_columns
            )
        )

        X_test = (
            X_test.reindex(
                columns=self.feature_columns
            )
        )


        self.build_preprocessor(
            X_train
        )

        X_train_processed = (
            self.preprocessor.fit_transform(
                X_train
            )
        )

        X_validation_processed = (
            self.preprocessor.transform(
                X_validation
            )
        )

        X_test_processed = (
            self.preprocessor.transform(
                X_test
            )
        )

        return {

            "train_df":
                train_df,

            "validation_df":
                validation_df,

            "test_df":
                test_df,



            "train_metadata":
                train_metadata,

            "validation_metadata":
                validation_metadata,

            "test_metadata":
                test_metadata,


            "X_train":
                X_train,

            "X_validation":
                X_validation,

            "X_test":
                X_test,


            "y_train":
                y_train,

            "y_validation":
                y_validation,

            "y_test":
                y_test,


            "X_train_processed":
                X_train_processed,

            "X_validation_processed":
                X_validation_processed,

            "X_test_processed":
                X_test_processed,


            "preprocessor":
                self.preprocessor
        }