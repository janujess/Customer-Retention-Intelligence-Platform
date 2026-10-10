from time import perf_counter
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import average_precision_score

from tqdm.auto import tqdm


class ModelTuner:

    def __init__(
        self,
        models,
        param_grids,
        n_splits,
        gap_periods,
        save_dir
    ):


        if not isinstance(models, dict):

            raise TypeError(
                "models must be a dictionary."
            )


        if not models:

            raise ValueError(
                "models dictionary cannot be empty."
            )


        if not isinstance(param_grids, dict):

            raise TypeError(
                "param_grids must be a dictionary."
            )


        if not param_grids:

            raise ValueError(
                "param_grids cannot be empty."
            )


        if n_splits < 2:

            raise ValueError(
                "n_splits must be at least 2."
            )


        if gap_periods < 0:

            raise ValueError(
                "gap_periods cannot be negative."
            )


        self.models = models

        self.param_grids = param_grids

        self.n_splits = n_splits

        self.gap_periods = gap_periods


        self.save_dir = Path(
            save_dir
        )


        self.save_dir.mkdir(
            parents=True,
            exist_ok=True


        self.results = None

        self.best_params = {}

        self.best_scores = {}

        self.best_models = {}


   #validate tuning data

    def validate_data(
        self,
        X_train,
        y_train,
        snapshot_dates
    ):

        if X_train is None:

            raise ValueError(
                "X_train is required."
            )


        if y_train is None:

            raise ValueError(
                "y_train is required."
            )


        if snapshot_dates is None:

            raise ValueError(
                "snapshot_dates are required."
            )


        if len(X_train) != len(y_train):

            raise ValueError(
                "X_train and y_train must "
                "have the same number of rows."
            )


        if len(X_train) != len(snapshot_dates):

            raise ValueError(
                "X_train and snapshot_dates must "
                "have the same number of rows."
            )


        if len(X_train) == 0:

            raise ValueError(
                "Training data is empty."
            )

        target_values = set(
            np.unique(
                np.asarray(
                    y_train
                ).astype(int)
            )
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

    def create_temporal_folds(
        self,
        snapshot_dates
    ):

        dates = pd.to_datetime(
            snapshot_dates
        )

        unique_dates = np.array(
            sorted(
                dates.unique()
            )
        )


        minimum_periods = (
            self.n_splits
            + self.gap_periods
            + 1
        )


        if len(unique_dates) < minimum_periods:

            raise ValueError(
                "Not enough unique snapshot dates "
                "for the requested temporal CV."
            )

        validation_size = (
            len(unique_dates)
            //
            (self.n_splits + 1)
        )


        if validation_size < 1:

            raise ValueError(
                "Validation window size is zero."
            )


        folds = []

        for fold_number in range(
            self.n_splits
        ):

            validation_start_position = (
                len(unique_dates)
                -
                (
                    self.n_splits
                    - fold_number
                )
                * validation_size
            )


            validation_end_position = (
                validation_start_position
                + validation_size
            )


            train_end_position = (
                validation_start_position
                -
                self.gap_periods
            )


            if train_end_position <= 0:

                continue


            train_dates = unique_dates[
                :train_end_position
            ]

            validation_dates = unique_dates[
                validation_start_position:
                validation_end_position
            ]

            train_indices = np.where(
                dates.isin(
                    train_dates
                )
            )[0]


            validation_indices = np.where(
                dates.isin(
                    validation_dates
                )
            )[0]


            if len(train_indices) == 0:

                continue


            if len(validation_indices) == 0:

                continue


            folds.append(
                (
                    train_indices,
                    validation_indices
                )
            )


        if len(folds) != self.n_splits:

            raise ValueError(
                "Could not create the requested "
                "number of temporal folds."
            )


        return folds

    def save_best_model(
        self,
        model_name,
        X_train,
        y_train,
        results_df
    ):

        best_parameters = (
            self.best_params[
                model_name
            ]
        )


        best_score = float(
            self.best_scores[
                model_name
            ]
        )

        best_model = clone(
            self.models[
                model_name
            ]
        )

        best_model.set_params(
            **best_parameters
        )

        print(
            f"\nRetraining best "
            f"{model_name} "
            f"on full training data..."
        )


        start_time = (
            perf_counter()
        )


        best_model.fit(
            X_train,
            y_train
        )


        retraining_seconds = (
            perf_counter()
            -
            start_time
        )


        print(
            f"Retraining completed in "
            f"{retraining_seconds:.2f} seconds."
        )

        model_path = (
            self.save_dir
            /
            f"best_{model_name}.joblib"
        )


        joblib.dump(
            best_model,
            model_path
        )

        results_path = (
            self.save_dir
            /
            f"{model_name}_tuning_results.csv"
        )


        results_df.to_csv(
            results_path,
            index=False
        )

        metadata = {

            "model":
                model_name,

            "best_cv_pr_auc":
                best_score,

            "best_parameters":
                best_parameters,

            "retraining_seconds":
                retraining_seconds,

            "n_splits":
                self.n_splits,

            "gap_periods":
                self.gap_periods
        }


        metadata_path = (
            self.save_dir
            /
            f"{model_name}_tuning_metadata.json"
        )


        with open(
            metadata_path,
            "w"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
                default=str
            )

        self.best_models[
            model_name
        ] = best_model


        print(
            f"\nBest model saved to:"
        )

        print(
            model_path
        )


        print(
            f"\nTuning results saved to:"
        )

        print(
            results_path
        )


        print(
            f"\nTuning metadata saved to:"
        )

        print(
            metadata_path
        )


        return best_model


    def tune_model(
        self,
        model_name,
        X_train,
        y_train,
        snapshot_dates
    ):
        if model_name not in self.models:

            raise ValueError(
                f"Model '{model_name}' "
                "was not supplied."
            )
        if model_name not in self.param_grids:

            raise ValueError(
                f"No parameter grid supplied "
                f"for '{model_name}'."
            )

        folds = (
            self.create_temporal_folds(
                snapshot_dates=
                    snapshot_dates
            )
        )

        parameter_combinations = list(
            ParameterGrid(
                self.param_grids[
                    model_name
                ]
            )
        )


        if not parameter_combinations:

            raise ValueError(
                f"No parameter combinations "
                f"were created for {model_name}."
            )

        print(
            f"\n{'=' * 60}"
        )


        print(
            f"TUNING: "
            f"{model_name.upper()}"
        )


        print(
            f"{'=' * 60}"
        )


        print(
            f"Parameter combinations: "
            f"{len(parameter_combinations):,}"
        )


        print(
            f"Temporal folds: "
            f"{len(folds)}"
        )


        print(
            f"Total model fits: "
            f"{len(parameter_combinations) * len(folds):,}"
        )


        model_results = []

        progress_bar = tqdm(

            parameter_combinations,

            desc=
                f"Tuning {model_name}",

            unit="config"
        )


        best_score_so_far = (
            -np.inf
        )

        for params in progress_bar:

            fold_scores = []


            start_time = (
                perf_counter()
            )

            for (
                fold_train_indices,
                fold_validation_indices
            ) in folds:
                # =============================================

                model = clone(
                    self.models[
                        model_name
                    ]
                )

                model.set_params(
                    **params
                )
                if isinstance(
                    X_train,
                    pd.DataFrame
                ):

                    X_fold_train = (
                        X_train.iloc[
                            fold_train_indices
                        ]
                    )

                else:

                    X_fold_train = (
                        X_train[
                            fold_train_indices
                        ]
                    )


                if isinstance(
                    y_train,
                    pd.Series
                ):

                    y_fold_train = (
                        y_train.iloc[
                            fold_train_indices
                        ]
                    )

                else:

                    y_fold_train = (
                        y_train[
                            fold_train_indices
                        ]
                    )

                if isinstance(
                    X_train,
                    pd.DataFrame
                ):

                    X_fold_validation = (
                        X_train.iloc[
                            fold_validation_indices
                        ]
                    )

                else:

                    X_fold_validation = (
                        X_train[
                            fold_validation_indices
                        ]
                    )


                if isinstance(
                    y_train,
                    pd.Series
                ):

                    y_fold_validation = (
                        y_train.iloc[
                            fold_validation_indices
                        ]
                    )

                else:

                    y_fold_validation = (
                        y_train[
                            fold_validation_indices
                        ]
                    )

                if (
                    len(
                        np.unique(
                            y_fold_train
                        )
                    ) < 2

                    or

                    len(
                        np.unique(
                            y_fold_validation
                        )
                    ) < 2
                ):

                    continue

                model.fit(
                    X_fold_train,
                    y_fold_train
                )

                probabilities = (
                    model.predict_proba(
                        X_fold_validation
                    )[:, 1]
                )

                fold_pr_auc = (
                    average_precision_score(
                        y_fold_validation,
                        probabilities
                    )
                )


                fold_scores.append(
                    fold_pr_auc
                )


            # =================================================
            # MAKE SURE FOLDS WERE EVALUATED
            # =================================================

            if not fold_scores:

                continue


            # =================================================
            # CALCULATE SUMMARY STATISTICS
            # =================================================

            mean_pr_auc = float(
                np.mean(
                    fold_scores
                )
            )


            std_pr_auc = float(
                np.std(
                    fold_scores
                )
            )


            min_pr_auc = float(
                np.min(
                    fold_scores
                )
            )


            max_pr_auc = float(
                np.max(
                    fold_scores
                )
            )


            elapsed_seconds = (
                perf_counter()
                -
                start_time
            )


            # =================================================
            # UPDATE BEST SCORE
            # =================================================

            if (
                mean_pr_auc
                >
                best_score_so_far
            ):

                best_score_so_far = (
                    mean_pr_auc
                )


            # =================================================
            # SAVE RESULT
            # =================================================

            model_results.append(
                {

                    "model":
                        model_name,

                    "parameters":
                        params,

                    "mean_pr_auc":
                        mean_pr_auc,

                    "std_pr_auc":
                        std_pr_auc,

                    "min_pr_auc":
                        min_pr_auc,

                    "max_pr_auc":
                        max_pr_auc,

                    "folds_used":
                        len(
                            fold_scores
                        ),

                    "training_seconds":
                        elapsed_seconds
                }
            )


            # =================================================
            # UPDATE PROGRESS BAR
            # =================================================

            progress_bar.set_postfix(
                {

                    "current":
                        f"{mean_pr_auc:.5f}",

                    "best":
                        f"{best_score_so_far:.5f}"
                }
            )


        # =====================================================
        # CHECK RESULTS
        # =====================================================

        if not model_results:

            raise ValueError(
                f"No valid tuning results "
                f"were produced for "
                f"{model_name}."
            )


        # =====================================================
        # CREATE RESULTS DATAFRAME
        # =====================================================

        results_df = (
            pd.DataFrame(
                model_results
            )
        )


        # =====================================================
        # SORT BY MEAN PR-AUC
        # =====================================================

        results_df = (
            results_df

            .sort_values(
                "mean_pr_auc",
                ascending=False
            )

            .reset_index(
                drop=True
            )
        )


        # =====================================================
        # SAVE BEST PARAMETERS
        # =====================================================

        self.best_params[
            model_name
        ] = results_df.loc[
            0,
            "parameters"
        ]


        # =====================================================
        # SAVE BEST SCORE
        # =====================================================

        self.best_scores[
            model_name
        ] = float(
            results_df.loc[
                0,
                "mean_pr_auc"
            ]
        )


        # =====================================================
        # PRINT BEST RESULT
        # =====================================================

        print(
            f"\nBest {model_name} "
            f"CV PR-AUC: "
            f"{self.best_scores[model_name]:.5f}"
        )


        print(
            "\nBest parameters:"
        )


        for parameter, value in (
            self.best_params[
                model_name
            ].items()
        ):

            print(
                f"  {parameter}: "
                f"{value}"
            )


        # =====================================================
        # RETRAIN AND SAVE BEST MODEL
        # =====================================================

        self.save_best_model(

            model_name=
                model_name,

            X_train=
                X_train,

            y_train=
                y_train,

            results_df=
                results_df
        )


        return results_df


    # =========================================================
    # 5. TUNE ALL SELECTED MODELS
    # =========================================================

    def tune(
        self,
        X_train,
        y_train,
        snapshot_dates
    ):

        # =====================================================
        # VALIDATE DATA
        # =====================================================

        self.validate_data(

            X_train=
                X_train,

            y_train=
                y_train,

            snapshot_dates=
                snapshot_dates
        )


        all_results = []


        # =====================================================
        # TUNE EACH SELECTED MODEL
        # =====================================================

        for model_name in self.models:

            results = (
                self.tune_model(

                    model_name=
                        model_name,

                    X_train=
                        X_train,

                    y_train=
                        y_train,

                    snapshot_dates=
                        snapshot_dates
                )
            )


            all_results.append(
                results
            )

        self.results = (
            pd.concat(
                all_results,
                ignore_index=True
            )
        )

        combined_results_path = (
            self.save_dir
            /
            "all_tuning_results.csv"
        )


        self.results.to_csv(
            combined_results_path,
            index=False
        )


        print(
            f"\nCombined tuning results "
            f"saved to:"
        )

        print(
            combined_results_path
        )


        return self.results