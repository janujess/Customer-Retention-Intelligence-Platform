from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


class ModelTester:

    def __init__(
        self,
        threshold,
        top_k,
        results_dir,
        id_column="account_id",
        date_column="snapshot_date"
    ):
        #validating settings

        if not 0 <= threshold <= 1:

            raise ValueError(
                "threshold must be between 0 and 1."
            )


        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )


        self.threshold = threshold

        self.top_k = top_k

        self.id_column = id_column

        self.date_column = date_column

        self.results_dir = Path(
            results_dir
        )


        self.results_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.results = None

        self.weekly_results = None

        self.confusion_matrices = {}

        self.prediction_results = {}



    def validate_inputs(
        self,
        models,
        X_test,
        y_test,
        test_metadata
    ):


        if not isinstance(
            models,
            dict
        ):

            raise TypeError(
                "models must be a dictionary."
            )


        if not models:

            raise ValueError(
                "models dictionary is empty."
            )


        if X_test is None:

            raise ValueError(
                "X_test is required."
            )


        if y_test is None:

            raise ValueError(
                "y_test is required."
            )


        if len(X_test) != len(
            y_test
        ):

            raise ValueError(
                "X_test and y_test must have "
                "the same number of rows."
            )


        if len(y_test) == 0:

            raise ValueError(
                "Test data is empty."
            )


        if test_metadata is None:

            raise ValueError(
                "test_metadata is required."
            )


        if len(test_metadata) != len(
            y_test
        ):

            raise ValueError(
                "test_metadata and y_test must have "
                "the same number of rows."
            )


        required_metadata_columns = [
            self.id_column,
            self.date_column
        ]


        missing_metadata_columns = [

            column

            for column
            in required_metadata_columns

            if column
            not in test_metadata.columns
        ]


        if missing_metadata_columns:

            raise ValueError(

                f"Missing metadata columns: "
                f"{missing_metadata_columns}"
            )


        target_values = set(

            pd.Series(
                y_test
            )

            .dropna()

            .astype(int)

            .unique()
        )


        if not target_values.issubset(
            {0, 1}
        ):

            raise ValueError(
                "y_test must contain only 0 and 1."
            )


        if len(target_values) < 2:

            raise ValueError(
                "Test data must contain both classes."
            )


#getting probabilities

    def get_probabilities(
        self,
        model,
        X_test
    ):

        if not hasattr(
            model,
            "predict_proba"
        ):

            raise ValueError(
                "Model does not support predict_proba()."
            )


        probabilities = (

            model.predict_proba(
                X_test
            )[:, 1]
        )
    #validate the probabilities
        if np.isnan(
            probabilities
        ).any():

            raise ValueError(
                "Model probabilities contain "
                "missing values."
            )


        if (
            (probabilities < 0).any()
            or
            (probabilities > 1).any()
        ):

            raise ValueError(
                "Model probabilities must be "
                "between 0 and 1."
            )


        return probabilities


    def get_predictions(
        self,
        probabilities
    ):

        predictions = (

            np.asarray(
                probabilities
            )

            >= self.threshold

        ).astype(int)


        return predictions

    #weekly top k metrics
    def calculate_weekly_top_k_metrics(
            self,
            y_true,
            probabilities,
            metadata,
            k=None
    ):

        if k is None:
            k = self.top_k

        # -----------------------------------------------------
        # VALIDATE K
        # -----------------------------------------------------

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        # -----------------------------------------------------
        # VALIDATE METADATA
        # -----------------------------------------------------

        if metadata is None:
            raise ValueError(
                "Metadata is required for "
                "weekly Top-K evaluation."
            )

        if len(metadata) != len(y_true):
            raise ValueError(
                "metadata and y_true must have "
                "the same number of rows."
            )

        if len(probabilities) != len(y_true):
            raise ValueError(
                "probabilities and y_true must have "
                "the same number of rows."
            )

        if (
                self.date_column
                not in metadata.columns
        ):
            raise ValueError(
                f"metadata must contain "
                f"'{self.date_column}'."
            )


        evaluation_df = (

            metadata
            .copy()
            .reset_index(
                drop=True
            )
        )

        evaluation_df[
            self.date_column
        ] = pd.to_datetime(

            evaluation_df[
                self.date_column
            ],

            errors="coerce"
        )

        if evaluation_df[
            self.date_column
        ].isna().any():
            raise ValueError(
                "Test metadata contains "
                "invalid snapshot dates."
            )


        evaluation_df[
            "actual"
        ] = np.asarray(
            y_true
        ).astype(int)


        evaluation_df[
            "probability"
        ] = np.asarray(
            probabilities
        )

        weekly_rows = []


        for (
                snapshot_date,
                weekly_df
        ) in evaluation_df.groupby(

            self.date_column,

            sort=True
        ):


            weekly_df = (

                weekly_df

                .sort_values(
                    "probability",
                    ascending=False
                )

                .reset_index(
                    drop=True
                )
            )


            eligible_customers = len(
                weekly_df
            )

            weekly_k = min(
                k,
                eligible_customers
            )

            top_k_df = weekly_df.head(
                weekly_k
            )

            actual_churners = int(

                weekly_df[
                    "actual"
                ].sum()
            )

            captured_churners = int(

                top_k_df[
                    "actual"
                ].sum()
            )

            precision_at_k = (

                captured_churners
                /
                weekly_k

                if weekly_k > 0

                else 0.0
            )


            recall_at_k = (

                captured_churners
                /
                actual_churners

                if actual_churners > 0

                else np.nan
            )



            weekly_churn_rate = (

                actual_churners
                /
                eligible_customers

                if eligible_customers > 0

                else 0.0
            )



            lift_at_k = (

                precision_at_k
                /
                weekly_churn_rate

                if weekly_churn_rate > 0

                else np.nan
            )


            percent_targeted = (

                weekly_k
                /
                eligible_customers
                * 100

                if eligible_customers > 0

                else 0.0
            )

            #store the results

            weekly_rows.append(
                {

                    "snapshot_date":
                        snapshot_date,

                    "eligible_customers":
                        eligible_customers,

                    "k_requested":
                        k,

                    "k_used":
                        weekly_k,

                    "percent_targeted":
                        percent_targeted,

                    "actual_churners":
                        actual_churners,

                    "captured_churners":
                        captured_churners,

                    f"precision_at_{k}":
                        precision_at_k,

                    f"recall_at_{k}":
                        recall_at_k,

                    f"lift_at_{k}":
                        lift_at_k,

                    "weekly_churn_rate":
                        weekly_churn_rate
                }
            )


        weekly_results = pd.DataFrame(
            weekly_rows
        )

        total_actual_churners = int(

            weekly_results[
                "actual_churners"
            ].sum()
        )

        total_captured_churners = int(

            weekly_results[
                "captured_churners"
            ].sum()
        )

        aggregate_recall = (

            total_captured_churners
            /
            total_actual_churners

            if total_actual_churners > 0

            else 0.0
        )

        mean_precision = float(

            weekly_results[
                f"precision_at_{k}"
            ].mean()
        )

        mean_recall = float(

            weekly_results[
                f"recall_at_{k}"
            ].mean(
                skipna=True
            )
        )

        mean_lift = float(

            weekly_results[
                f"lift_at_{k}"
            ].mean(
                skipna=True
            )
        )

        mean_percent_targeted = float(

            weekly_results[
                "percent_targeted"
            ].mean()
        )

        summary = {

            "top_k":
                k,

            "weeks_evaluated":
                len(
                    weekly_results
                ),

            "total_actual_churners":
                total_actual_churners,

            "total_captured_churners":
                total_captured_churners,

            f"aggregate_recall_at_{k}":
                aggregate_recall,

            f"mean_precision_at_{k}":
                mean_precision,

            f"mean_recall_at_{k}":
                mean_recall,

            f"mean_lift_at_{k}":
                mean_lift,

            "mean_percent_targeted":
                mean_percent_targeted
        }

        return (
            summary,
            weekly_results
        )

        for (
            snapshot_date,
            weekly_df
        ) in evaluation_df.groupby(

            self.date_column,
            sort=True
        ):

            weekly_df = (

                weekly_df

                .sort_values(
                    "probability",
                    ascending=False
                )

                .reset_index(
                    drop=True
                )
            )


            eligible_customers = len(
                weekly_df
            )


            weekly_k = min(
                self.top_k,
                eligible_customers
            )


            top_k_df = weekly_df.head(
                weekly_k
            )

            actual_churners = int(

                weekly_df[
                    "actual"
                ].sum()
            )

            captured_churners = int(

                top_k_df[
                    "actual"
                ].sum()
            )

            precision_at_k = (

                captured_churners
                /
                weekly_k

                if weekly_k > 0

                else 0.0
            )

            recall_at_k = (

                captured_churners
                /
                actual_churners

                if actual_churners > 0

                else np.nan
            )

            weekly_churn_rate = (

                actual_churners
                /
                eligible_customers

                if eligible_customers > 0

                else 0.0
            )

            lift_at_k = (

                precision_at_k
                /
                weekly_churn_rate

                if weekly_churn_rate > 0

                else np.nan
            )

            weekly_rows.append(
                {

                    "snapshot_date":
                        snapshot_date,

                    "eligible_customers":
                        eligible_customers,

                    "k_used":
                        weekly_k,

                    "actual_churners":
                        actual_churners,

                    "captured_churners":
                        captured_churners,

                    f"precision_at_{self.top_k}":
                        precision_at_k,

                    f"recall_at_{self.top_k}":
                        recall_at_k,

                    f"lift_at_{self.top_k}":
                        lift_at_k,

                    "weekly_churn_rate":
                        weekly_churn_rate
                }
            )


        weekly_results = pd.DataFrame(
            weekly_rows
        )

        total_actual_churners = int(

            weekly_results[
                "actual_churners"
            ].sum()
        )


        total_captured_churners = int(

            weekly_results[
                "captured_churners"
            ].sum()
        )


        aggregate_recall = (

            total_captured_churners
            /
            total_actual_churners

            if total_actual_churners > 0

            else 0.0
        )


        summary = {

            "weeks_evaluated":
                len(
                    weekly_results
                ),

            "total_actual_churners":
                total_actual_churners,

            "total_captured_churners":
                total_captured_churners,

            f"aggregate_recall_at_{self.top_k}":
                aggregate_recall,

            f"mean_precision_at_{self.top_k}":
                float(
                    weekly_results[
                        f"precision_at_{self.top_k}"
                    ].mean()
                ),

            f"mean_recall_at_{self.top_k}":
                float(
                    weekly_results[
                        f"recall_at_{self.top_k}"
                    ].mean(
                        skipna=True
                    )
                ),

            f"mean_lift_at_{self.top_k}":
                float(
                    weekly_results[
                        f"lift_at_{self.top_k}"
                    ].mean(
                        skipna=True
                    )
                )
        }


        return (
            summary,
            weekly_results
        )

    def test_model(
        self,
        model,
        model_name,
        X_test,
        y_test,
        test_metadata
    ):


        y_true = (

            np.asarray(
                y_test
            )

            .astype(int)
        )

        probabilities = (

            self.get_probabilities(

                model=model,

                X_test=X_test
            )
        )

        predictions = (

            self.get_predictions(
                probabilities
            )
        )

        accuracy = accuracy_score(
            y_true,
            predictions
        )


        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )


        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )


        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )


        pr_auc = (
            average_precision_score(
                y_true,
                probabilities
            )
        )


        brier_score = (
            brier_score_loss(
                y_true,
                probabilities
            )
        )


        prevalence = float(
            y_true.mean()
        )


        pr_auc_lift = (

            pr_auc
            /
            prevalence

            if prevalence > 0

            else np.nan
        )

        cm = confusion_matrix(

            y_true,

            predictions,

            labels=[
                0,
                1
            ]
        )


        tn, fp, fn, tp = (
            cm.ravel()
        )


        self.confusion_matrices[
            model_name
        ] = cm

        prediction_df = (

            test_metadata
            .copy()
            .reset_index(drop=True)
        )


        prediction_df[
            "actual"
        ] = y_true


        prediction_df[
            "probability"
        ] = probabilities


        prediction_df[
            "prediction"
        ] = predictions


        self.prediction_results[
            model_name
        ] = prediction_df



        (
            weekly_summary,
            weekly_results
        ) = (

            self.calculate_weekly_top_k_metrics(

                y_true=y_true,

                probabilities=probabilities,

                metadata=test_metadata
            )
        )


        weekly_results[
            "model"
        ] = model_name


        result = {

            "model":
                model_name,

            "threshold":
                self.threshold,

            "rows":
                len(
                    y_true
                ),

            "positive_rows":
                int(
                    y_true.sum()
                ),

            "prevalence":
                prevalence,

            "pr_auc":
                pr_auc,

            "pr_auc_lift":
                pr_auc_lift,

            "roc_auc":
                roc_auc,

            "brier_score":
                brier_score,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "true_positives":
                int(
                    tp
                ),

            "false_positives":
                int(
                    fp
                ),

            "false_negatives":
                int(
                    fn
                ),

            "true_negatives":
                int(
                    tn
                )
        }


        result.update(
            weekly_summary
        )


        return (
            result,
            weekly_results
        )

    def test(
        self,
        models,
        X_test,
        y_test,
        test_metadata
    ):


        self.validate_inputs(

            models=models,

            X_test=X_test,

            y_test=y_test,

            test_metadata=test_metadata
        )
        self.confusion_matrices = {}

        self.prediction_results = {}


        test_results = []

        weekly_result_frames = []


        for (
            model_name,
            model
        ) in models.items():


            print(
                f"Testing {model_name}..."
            )


            (
                model_results,
                weekly_results
            ) = (

                self.test_model(

                    model=model,

                    model_name=model_name,

                    X_test=X_test,

                    y_test=y_test,

                    test_metadata=test_metadata
                )
            )


            test_results.append(
                model_results
            )


            weekly_result_frames.append(
                weekly_results
            )


        self.results = (

            pd.DataFrame(
                test_results
            )

            .sort_values(
                "pr_auc",
                ascending=False
            )

            .reset_index(
                drop=True
            )
        )

        self.weekly_results = (

            pd.concat(
                weekly_result_frames,
                ignore_index=True
            )
        )

        self.save_test_results()


        return self.results


    def get_confusion_matrix(
        self,
        model_name,
        include_percentages=True
    ):

        if (
            model_name
            not in self.confusion_matrices
        ):

            raise ValueError(
                f"No confusion matrix found "
                f"for '{model_name}'."
            )


        cm = (
            self.confusion_matrices[
                model_name
            ]
        )


        cm_df = pd.DataFrame(

            cm,

            index=[
                "Actual Non-Churn",
                "Actual Churn"
            ],

            columns=[
                "Predicted Non-Churn",
                "Predicted Churn"
            ]
        )


        if not include_percentages:

            return cm_df


        row_totals = cm.sum(
            axis=1,
            keepdims=True
        )


        percentages = np.divide(

            cm,

            row_totals,

            out=np.zeros_like(
                cm,
                dtype=float
            ),

            where=
                row_totals != 0

        ) * 100


        percentage_df = pd.DataFrame(

            percentages,

            index=cm_df.index,

            columns=cm_df.columns
        )


        return (
            cm_df,
            percentage_df
        )

    def plot_confusion_matrix(
        self,
        model_name,
        save_path=None,
        show=True
    ):

        if (
            model_name
            not in self.confusion_matrices
        ):

            raise ValueError(
                f"No confusion matrix found "
                f"for '{model_name}'."
            )


        cm = (
            self.confusion_matrices[
                model_name
            ]
        )

        row_totals = cm.sum(
            axis=1,
            keepdims=True
        )


        percentages = np.divide(

            cm,

            row_totals,

            out=np.zeros_like(
                cm,
                dtype=float
            ),

            where=
                row_totals != 0

        ) * 100

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )


        image = ax.imshow(

            percentages,

            vmin=0,

            vmax=100
        )


        for row in range(
            cm.shape[0]
        ):

            for column in range(
                cm.shape[1]
            ):


                ax.text(

                    column,

                    row,

                    (
                        f"{cm[row, column]:,}"
                        f"\n"
                        f"{percentages[row, column]:.1f}%"
                    ),

                    ha="center",

                    va="center",

                    fontsize=12
                )

        ax.set_xticks(
            [0, 1]
        )


        ax.set_xticklabels(
            [
                "Predicted Non-Churn",
                "Predicted Churn"
            ]
        )


        ax.set_yticks(
            [0, 1]
        )


        ax.set_yticklabels(
            [
                "Actual Non-Churn",
                "Actual Churn"
            ]
        )


        ax.set_xlabel(
            "Predicted Class"
        )


        ax.set_ylabel(
            "Actual Class"
        )


        ax.set_title(

            f"{model_name}\n"

            f"Final Test Confusion Matrix | "

            f"Threshold = "
            f"{self.threshold:.2f}"
        )


        colorbar = fig.colorbar(
            image,
            ax=ax
        )


        colorbar.set_label(
            "% within actual class"
        )


        plt.tight_layout()

        if save_path is not None:

            save_path = Path(
                save_path
            )


            save_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )


        if show:

            plt.show()

        else:

            plt.close(
                fig
            )


        return fig

    def show_all_confusion_matrices(
        self
    ):

        if not self.confusion_matrices:

            raise ValueError(
                "No confusion matrices are available."
            )


        for model_name in (
            self.confusion_matrices
        ):


            self.plot_confusion_matrix(

                model_name=model_name,

                show=True
            )

    def save_test_results(
        self
    ):


        if self.results is None:

            raise ValueError(
                "No test results are available."
            )


        if self.weekly_results is None:

            raise ValueError(
                "No weekly test results are available."
            )


        # =====================================================
        # DIRECTORIES
        # =====================================================

        predictions_dir = (

            self.results_dir
            /
            "predictions"
        )


        confusion_dir = (

            self.results_dir
            /
            "confusion_matrices"
        )


        predictions_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        confusion_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        metrics_path = (

            self.results_dir
            /
            "test_metrics.csv"
        )


        self.results.to_csv(

            metrics_path,

            index=False
        )

        weekly_path = (

            self.results_dir
            /
            f"weekly_top_{self.top_k}_metrics.csv"
        )


        self.weekly_results.to_csv(

            weekly_path,

            index=False
        )
        for (
            model_name,
            prediction_df
        ) in self.prediction_results.items():


            prediction_path = (

                predictions_dir
                /
                f"{model_name}_test_predictions.csv"
            )


            prediction_df.to_csv(

                prediction_path,

                index=False
            )

        for model_name in (
            self.confusion_matrices
        ):


            confusion_path = (

                confusion_dir
                /
                f"{model_name}_test_confusion_matrix.png"
            )


            self.plot_confusion_matrix(

                model_name=model_name,

                save_path=confusion_path,

                show=False
            )

        print(
            "\n"
            + "=" * 60
        )


        print(
            "FINAL TEST RESULTS SAVED"
        )


        print(
            "=" * 60
        )


        print(
            f"\nModels tested: "
            f"{len(self.results)}"
        )


        print(
            f"Prediction files saved: "
            f"{len(self.prediction_results)}"
        )


        print(
            f"Confusion matrices saved: "
            f"{len(self.confusion_matrices)}"
        )


        print(
            "\nResults directory:"
        )


        print(
            self.results_dir
        )


        print(
            "\nOverall test metrics:"
        )


        print(
            metrics_path
        )


        print(
            f"\nWeekly Top-{self.top_k} metrics:"
        )


        print(
            weekly_path
        )