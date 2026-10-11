from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class ChurnPredictor:

    def __init__(
        self,
        engine,
        model,
        preprocessor,
        snapshot_builder,
        feature_engineer,
        feature_columns
    ):

        self.engine = engine

        self.model = model

        self.preprocessor = preprocessor

        self.snapshot_builder = snapshot_builder

        self.feature_engineer = feature_engineer

        self.feature_columns = list(
            feature_columns
        )

    @classmethod
    def from_files(
        cls,
        engine,
        model_path,
        preprocessor_path,
        feature_columns_path,
        snapshot_builder,
        feature_engineer
    ):

        model_path = Path(
            model_path
        )

        preprocessor_path = Path(
            preprocessor_path
        )

        feature_columns_path = Path(
            feature_columns_path
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found: "
                f"{model_path}"
            )


        if not preprocessor_path.exists():

            raise FileNotFoundError(
                f"Preprocessor not found: "
                f"{preprocessor_path}"
            )


        if not feature_columns_path.exists():

            raise FileNotFoundError(
                f"Feature columns not found: "
                f"{feature_columns_path}"
            )


        model = joblib.load(
            model_path
        )

        preprocessor = joblib.load(
            preprocessor_path
        )

        feature_columns = joblib.load(
            feature_columns_path
        )

        if not hasattr(
            model,
            "predict_proba"
        ):

            raise ValueError(
                "Loaded model does not support "
                "predict_proba()."
            )


        return cls(
            engine=engine,
            model=model,
            preprocessor=preprocessor,
            snapshot_builder=snapshot_builder,
            feature_engineer=feature_engineer,
            feature_columns=feature_columns
        )

    def validate_prediction_date(
        self,
        prediction_date
    ):

        prediction_date = pd.to_datetime(
            prediction_date
        ).normalize()


        if pd.isna(
            prediction_date
        ):

            raise ValueError(
                "prediction_date is invalid."
            )


        return prediction_date

    def build_prediction_snapshot(
        self,
        prediction_date
    ):

        prediction_date = (
            self.validate_prediction_date(
                prediction_date
            )
        )

        snapshots = (
            self.snapshot_builder
            .build_snapshots(
                start_date=prediction_date,
                end_date=prediction_date,
                frequency="7D"
            )
        )


        if snapshots.empty:

            raise ValueError(
                "No active customers were found "
                "for the selected prediction date."
            )


        return snapshots

    def build_prediction_features(
        self,
        snapshots
    ):

        if snapshots is None:

            raise ValueError(
                "snapshots are required."
            )


        if snapshots.empty:

            raise ValueError(
                "snapshots cannot be empty."
            )

        features = (
            self.feature_engineer
            .build_all_features(
                snapshots_df=snapshots,
                target_column=None
            )
        )


        if features.empty:

            raise ValueError(
                "Feature engineering returned "
                "no prediction rows."
            )


        return features

    def validate_feature_schema(
        self,
        features_df
    ):

        missing_features = [

            column

            for column in self.feature_columns

            if column not in features_df.columns
        ]


        if missing_features:

            raise ValueError(
                "Prediction data is missing "
                "required model features: "
                f"{missing_features}"
            )


        return True

    def preprocess_features(
        self,
        features_df
    ):

        self.validate_feature_schema(
            features_df
        )

        X = features_df[
            self.feature_columns
        ].copy()

        X_processed = (
            self.preprocessor
            .transform(X)
        )


        return X_processed

    def predict_probabilities(
        self,
        X_processed
    ):

        probabilities = (
            self.model
            .predict_proba(
                X_processed
            )[:, 1]
        )


        return probabilities

    def apply_threshold(
        self,
        probabilities,
        threshold
    ):

        if not 0 <= threshold <= 1:

            raise ValueError(
                "threshold must be between "
                "0 and 1."
            )


        predictions = (
            probabilities
            >= threshold
        ).astype(int)


        return predictions

    def create_prediction_results(
        self,
        features_df,
        probabilities,
        threshold,
        prediction_date
    ):

        predictions = (
            self.apply_threshold(
                probabilities=probabilities,
                threshold=threshold
            )
        )


        results = pd.DataFrame({

            "account_id":
                features_df[
                    "account_id"
                ].to_numpy(),

            "prediction_date":
                pd.to_datetime(
                    prediction_date
                ),

            "churn_probability":
                probabilities,

            "churn_prediction":
                predictions
        })

        results = (
            results
            .sort_values(
                "churn_probability",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        results[
            "risk_rank"
        ] = np.arange(
            1,
            len(results) + 1
        )

        results[
            "threshold_used"
        ] = threshold


        results = results[
            [
                "account_id",
                "prediction_date",
                "risk_rank",
                "churn_probability",
                "churn_prediction",
                "threshold_used"
            ]
        ]
        return results



    def predict(
        self,
        prediction_date,
        threshold
    ):


        prediction_date = (
            self.validate_prediction_date(
                prediction_date
            )
        )

        snapshots = (
            self.build_prediction_snapshot(
                prediction_date
            )
        )

        features = (
            self.build_prediction_features(
                snapshots
            )
        )

        X_processed = (
            self.preprocess_features(
                features
            )
        )

        probabilities = (
            self.predict_probabilities(
                X_processed
            )
        )

        results = (
            self.create_prediction_results(
                features_df=features,
                probabilities=probabilities,
                threshold=threshold,
                prediction_date=prediction_date
            )
        )


        return results


    def get_at_risk_customers(
        self,
        predictions_df
    ):

        at_risk = (
            predictions_df[
                predictions_df[
                    "churn_prediction"
                ] == 1
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        return at_risk

    def get_top_risk_customers(
        self,
        predictions_df,
        top_k=100
    ):

        if top_k < 1:

            raise ValueError(
                "top_k must be at least 1."
            )


        top_k = min(
            top_k,
            len(predictions_df)
        )


        return (
            predictions_df
            .sort_values(
                "churn_probability",
                ascending=False
            )
            .head(
                top_k
            )
            .reset_index(
                drop=True
            )
        )

    def prediction_summary(
        self,
        predictions_df
    ):

        if predictions_df.empty:

            raise ValueError(
                "predictions_df cannot be empty."
            )


        customers_scored = len(
            predictions_df
        )


        customers_at_risk = int(
            predictions_df[
                "churn_prediction"
            ].sum()
        )


        percentage_at_risk = (
            customers_at_risk
            /
            customers_scored
            *
            100
        )


        summary = {

            "customers_scored":
                customers_scored,

            "customers_at_risk":
                customers_at_risk,

            "percentage_at_risk":
                percentage_at_risk,

            "average_churn_probability":
                predictions_df[
                    "churn_probability"
                ].mean(),

            "maximum_churn_probability":
                predictions_df[
                    "churn_probability"
                ].max(),

            "threshold_used":
                predictions_df[
                    "threshold_used"
                ].iloc[0]
        }


        return pd.DataFrame(
            [summary]
        )