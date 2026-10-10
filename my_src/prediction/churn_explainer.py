import numpy as np
import pandas as pd
import shap


class ChurnExplainer:

    def __init__(
        self,
        model,
        feature_names,
        feature_name_mapping=None
    ):

        self.model = model

        self.feature_names = list(
            feature_names
        )

        self.feature_name_mapping = (
            feature_name_mapping
            if feature_name_mapping is not None
            else {}
        )

        self.explainer = None

    def build_explainer(
        self,
        background_data=None
    ):

        try:

            self.explainer = (
                shap.TreeExplainer(
                    self.model
                )
            )

        except Exception:

            if background_data is None:

                raise ValueError(
                    "background_data is required "
                    "for this model type."
                )

            self.explainer = (
                shap.Explainer(
                    self.model,
                    background_data
                )
            )

        return self.explainer

    def calculate_shap_values(
        self,
        X
    ):

        if self.explainer is None:

            raise ValueError(
                "SHAP explainer has not been built."
            )

        shap_values = (
            self.explainer(X)
        )

        return shap_values


    # =========================================================
    # 3. EXPLAIN ONE CUSTOMER
    # =========================================================

    def explain_customer(
        self,
        X_customer,
        top_n=5
    ):

        if len(X_customer) != 1:

            raise ValueError(
                "X_customer must contain "
                "exactly one customer."
            )

        shap_values = (
            self.calculate_shap_values(
                X_customer
            )
        )

        values = np.asarray(
            shap_values.values
        )

        # Some binary classifiers may return
        # an additional class dimension.
        if values.ndim == 3:

            values = values[
                0,
                :,
                -1
            ]

        else:

            values = values[0]


        explanation = pd.DataFrame({

            "feature":
                self.feature_names,

            "feature_value":
                np.asarray(
                    X_customer
                )[0],

            "shap_value":
                values
        })

        explanation[
            "direction"
        ] = np.where(
            explanation[
                "shap_value"
            ] > 0,
            "increases_risk",
            "reduces_risk"
        )

        explanation[
            "absolute_impact"
        ] = (
            explanation[
                "shap_value"
            ].abs()
        )

        explanation[
            "display_name"
        ] = (
            explanation[
                "feature"
            ].map(
                self.feature_name_mapping
            )
            .fillna(
                explanation[
                    "feature"
                ]
            )
        )
        explanation = (
            explanation
            .sort_values(
                "absolute_impact",
                ascending=False
            )
            .head(
                top_n
            )
            .reset_index(
                drop=True
            )
        )


        return explanation

    def explain_customers(
        self,
        X,
        account_ids,
        top_n=5
    ):

        if len(X) != len(account_ids):

            raise ValueError(
                "X and account_ids must contain "
                "the same number of rows."
            )


        explanations = []


        for row_number, account_id in enumerate(
            account_ids
        ):

            X_customer = X[
                row_number:
                row_number + 1
            ]


            customer_explanation = (
                self.explain_customer(
                    X_customer=X_customer,
                    top_n=top_n
                )
            )


            customer_explanation.insert(
                0,
                "account_id",
                account_id
            )


            customer_explanation.insert(
                1,
                "factor_rank",
                np.arange(
                    1,
                    len(
                        customer_explanation
                    ) + 1
                )
            )


            explanations.append(
                customer_explanation
            )


        return pd.concat(
            explanations,
            ignore_index=True
        )