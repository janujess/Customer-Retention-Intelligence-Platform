from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


class ModelEvaluator:

    def __init__(
        self,
        threshold,
        results_dir
    ):
        if not 0 <= threshold <= 1:

            raise ValueError(
                "threshold must be between 0 and 1."
            )


        self.threshold = threshold

        self.results_dir = Path(
            results_dir
        )


        self.results_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.results = None

        self.confusion_matrices = {}

        self.prediction_results = {}

    def validate_inputs(
        self,
        models,
        X_validation,
        y_validation
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


        if X_validation is None:

            raise ValueError(
                "X_validation is required."
            )


        if y_validation is None:

            raise ValueError(
                "y_validation is required."
            )


        if len(X_validation) != len(
            y_validation
        ):

            raise ValueError(
                "X_validation and y_validation "
                "must have the same number of rows."
            )


        if len(y_validation) == 0:

            raise ValueError(
                "Validation data is empty."
            )

        target_values = set(
            np.unique(
                np.asarray(
                    y_validation
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
                "Validation data must contain "
                "both target classes."
            )

    def evaluate_model(
        self,
        model,
        model_name,
        X_validation,
        y_validation
    ):

        if not hasattr(
            model,
            "predict_proba"
        ):

            raise ValueError(
                f"{model_name} does not support "
                "predict_proba()."
            )

        probabilities = (

            model.predict_proba(
                X_validation
            )[:, 1]
        )

        if np.isnan(
            probabilities
        ).any():

            raise ValueError(
                f"{model_name} produced "
                "missing probabilities."
            )


        if (
            (probabilities < 0).any()
            or
            (probabilities > 1).any()
        ):

            raise ValueError(
                f"{model_name} produced "
                "invalid probabilities."
            )

        predictions = (

            probabilities
            >= self.threshold

        ).astype(int)


        cm = confusion_matrix(

            y_validation,

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


        prediction_df = pd.DataFrame(
            {

                "actual":
                    np.asarray(
                        y_validation
                    ).astype(int),

                "probability":
                    probabilities,

                "prediction":
                    predictions
            }
        )


        self.prediction_results[
            model_name
        ] = prediction_df

        pr_auc = (
            average_precision_score(

                y_validation,

                probabilities
            )
        )


        roc_auc = (
            roc_auc_score(

                y_validation,

                probabilities
            )
        )


        precision = (
            precision_score(

                y_validation,

                predictions,

                zero_division=0
            )
        )


        recall = (
            recall_score(

                y_validation,

                predictions,

                zero_division=0
            )
        )


        f1 = (
            f1_score(

                y_validation,

                predictions,

                zero_division=0
            )
        )


        prevalence = float(

            np.asarray(
                y_validation
            ).mean()
        )


        pr_auc_lift = (

            pr_auc
            /
            prevalence

            if prevalence > 0

            else np.nan
        )


        results = {

            "model":
                model_name,

            "threshold":
                self.threshold,

            "rows":
                len(
                    y_validation
                ),

            "positive_rows":
                int(
                    np.asarray(
                        y_validation
                    ).sum()
                ),

            "prevalence":
                prevalence,

            "pr_auc":
                pr_auc,

            "pr_auc_lift":
                pr_auc_lift,

            "roc_auc":
                roc_auc,

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


        return results


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

            f"Validation Confusion Matrix | "

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
                "No confusion matrices "
                "are available."
            )


        for model_name in (
            self.confusion_matrices
        ):


            self.plot_confusion_matrix(

                model_name=
                    model_name,

                show=True
            )


    def save_evaluation(
        self
    ):

        if self.results is None:

            raise ValueError(
                "No evaluation results "
                "are available."
            )


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
            "validation_metrics.csv"
        )


        self.results.to_csv(

            metrics_path,

            index=False
        )


        for (
            model_name,
            prediction_df
        ) in self.prediction_results.items():


            prediction_path = (

                predictions_dir
                /
                f"{model_name}_predictions.csv"
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
                f"{model_name}_confusion_matrix.png"
            )


            self.plot_confusion_matrix(

                model_name=
                    model_name,

                save_path=
                    confusion_path,

                show=False
            )

        print(
            "\n"
            + "=" * 60
        )


        print(
            "VALIDATION RESULTS SAVED"
        )


        print(
            "=" * 60
        )


        print(
            f"\nModels evaluated: "
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
            "\nMetrics:"
        )


        print(
            metrics_path
        )


    def evaluate(
        self,
        models,
        X_validation,
        y_validation
    ):


        self.validate_inputs(

            models=
                models,

            X_validation=
                X_validation,

            y_validation=
                y_validation
        )


        self.confusion_matrices = {}

        self.prediction_results = {}


        evaluation_results = []


        for (
            model_name,
            model
        ) in models.items():


            print(
                f"Evaluating "
                f"{model_name}..."
            )


            model_results = (

                self.evaluate_model(

                    model=
                        model,

                    model_name=
                        model_name,

                    X_validation=
                        X_validation,

                    y_validation=
                        y_validation
                )
            )


            evaluation_results.append(
                model_results
            )


        self.results = (

            pd.DataFrame(
                evaluation_results
            )

            .sort_values(
                "pr_auc",
                ascending=False
            )

            .reset_index(
                drop=True
            )
        )


        self.save_evaluation()


        return self.results