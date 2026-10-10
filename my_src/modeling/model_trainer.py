from time import perf_counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


class ModelTrainer:

    def __init__(
        self,
        model_params,
        save_dir
    ):

        if not isinstance(model_params, dict):

            raise TypeError(
                "model_params must be a dictionary."
            )

        if not model_params:

            raise ValueError(
                "model_params cannot be empty."
            )


        self.model_params = model_params

        self.save_dir = Path(
            save_dir
        )

        # Create directory if it does not exist
        self.save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.models = {}

        self.training_summary = None


    def validate_training_data(
        self,
        X_train,
        y_train
    ):

        if X_train is None or y_train is None:

            raise ValueError(
                "X_train and y_train are required."
            )


        if len(X_train) != len(y_train):

            raise ValueError(
                "X_train and y_train must have "
                "the same number of rows."
            )


        if len(y_train) == 0:

            raise ValueError(
                "Training data is empty."
            )


        y_array = np.asarray(
            y_train
        ).astype(int)


        target_values = set(
            np.unique(y_array)
        )


        if not target_values.issubset(
            {0, 1}
        ):

            raise ValueError(
                "Target must contain only 0 and 1."
            )


        if len(target_values) < 2:

            raise ValueError(
                "Training data must contain "
                "both target classes."
            )


        return {

            "rows":
                len(y_array),

            "positive_rows":
                int(
                    y_array.sum()
                ),

            "negative_rows":
                int(
                    len(y_array)
                    - y_array.sum()
                ),

            "positive_rate":
                float(
                    y_array.mean()
                )
        }

    def build_models(self):

        supported_models = {

            "logistic_regression":
                LogisticRegression,

            "random_forest":
                RandomForestClassifier,

            "xgboost":
                XGBClassifier,

            "catboost":
                CatBoostClassifier
        }


        models = {}


        for (
            model_name,
            parameters
        ) in self.model_params.items():


            if model_name not in supported_models:

                raise ValueError(
                    f"Unsupported model: "
                    f"{model_name}"
                )


            if not isinstance(
                parameters,
                dict
            ):

                raise TypeError(
                    f"Parameters for "
                    f"'{model_name}' "
                    f"must be a dictionary."
                )


            model_class = (
                supported_models[
                    model_name
                ]
            )


            models[
                model_name
            ] = model_class(
                **parameters
            )


        return models

    def save_model(
        self,
        model,
        model_name
    ):

        model_path = (
            self.save_dir
            /
            f"{model_name}.joblib"
        )


        joblib.dump(
            model,
            model_path
        )


        return model_path

    def train(
        self,
        X_train,
        y_train
    ):

        training_info = (
            self.validate_training_data(
                X_train,
                y_train
            )
        )


        print(
            f"Training rows: "
            f"{training_info['rows']:,}"
        )

        print(
            f"Churners: "
            f"{training_info['positive_rows']:,}"
        )

        print(
            f"Churn rate: "
            f"{training_info['positive_rate']:.2%}"
        )


        models = self.build_models()

        training_results = []


        for (
            model_name,
            model
        ) in models.items():


            print(
                f"\nTraining "
                f"{model_name}..."
            )


            start_time = (
                perf_counter()
            )

            model.fit(
                X_train,
                y_train
            )


            training_seconds = (
                perf_counter()
                - start_time
            )

            model_path = (
                self.save_model(
                    model=model,
                    model_name=model_name
                )
            )

            training_results.append(
                {

                    "model":
                        model_name,

                    "training_seconds":
                        training_seconds,

                    "model_path":
                        str(model_path)
                }
            )


            print(
                f"Completed in "
                f"{training_seconds:.2f} "
                f"seconds."
            )

            print(
                f"Saved to: "
                f"{model_path}"
            )


        self.models = models


        self.training_summary = (
            pd.DataFrame(
                training_results
            )
        )


        return self.models