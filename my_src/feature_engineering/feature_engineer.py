import numpy as np
import pandas as pd


class FeatureEngineer:

    def __init__(self, engine):

        self.engine = engine

    #helpers

    @staticmethod
    def safe_sum(series):

        return (
            pd.to_numeric(
                series,
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )


    @staticmethod
    def percentage_change(
        current_value,
        previous_value
    ):

        if previous_value > 0:

            return (
                current_value
                - previous_value
            ) / previous_value

        return np.nan


    @staticmethod
    def log_change(
        current_value,
        previous_value
    ):

        return (
            np.log1p(current_value)
            - np.log1p(previous_value)
        )


    @staticmethod
    def convert_binary_series(series):
        normalized = (
            series
            .astype("string")
            .str.strip()
            .str.lower()
        )

        binary_mapping = {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "1": 1,
            "0": 0
        }

        converted_text = (
            normalized.map(
                binary_mapping
            )
        )

        converted_numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        return (
            converted_text
            .fillna(converted_numeric)
            .astype("float64")
        )


   #load accounts from db

    def load_accounts(self):

        query = "SELECT account_id, industry, country, referral_source, signup_date FROM raw_accounts;"

        accounts = pd.read_sql(
            query,
            self.engine
        )

        accounts["signup_date"] = pd.to_datetime(
            accounts["signup_date"],
            errors="coerce"
        )

        accounts = (
            accounts
            .dropna(
                subset=["account_id"]
            )
            .drop_duplicates(
                subset=["account_id"],
                keep="last"
            )
        )

        return accounts

    #load subscription

    def load_subscriptions(self):

        query = """
        SELECT
            subscription_id,
            account_id,
            start_date,
            end_date,
            plan_tier,
            seats,
            mrr_amount,
            arr_amount,
            is_trial,
            upgrade_flag,
            downgrade_flag,
            churn_flag,
            billing_frequency,
            auto_renew_flag
        FROM raw_subscriptions;
        """

        subscriptions = pd.read_sql(
            query,
            self.engine
        )

        subscriptions["start_date"] = pd.to_datetime(
            subscriptions["start_date"],
            errors="coerce"
        )

        subscriptions["end_date"] = pd.to_datetime(
            subscriptions["end_date"],
            errors="coerce"
        )

        numeric_columns = [
            "seats",
            "mrr_amount",
            "arr_amount"
        ]

        for column in numeric_columns:

            subscriptions[column] = pd.to_numeric(
                subscriptions[column],
                errors="coerce"
            )

        binary_columns = [
            "is_trial",
            "upgrade_flag",
            "downgrade_flag",
            "churn_flag",
            "auto_renew_flag"
        ]

        for column in binary_columns:

            subscriptions[column] = (
                self.convert_binary_series(
                    subscriptions[column]
                )
            )

        subscriptions = (
            subscriptions
            .dropna(
                subset=[
                    "subscription_id",
                    "account_id",
                    "start_date"
                ]
            )
            .drop_duplicates(
                subset=["subscription_id"],
                keep="last"
            )
        )

        return subscriptions


    #load usage data from db

    def load_usage(self):

        query = """
        SELECT
            usage_id,
            subscription_id,
            usage_date,
            feature_name,
            usage_count,
            usage_duration_secs,
            error_count
        FROM raw_feature_usage;
        """

        usage = pd.read_sql(
            query,
            self.engine
        )

        usage["usage_date"] = pd.to_datetime(
            usage["usage_date"],
            errors="coerce"
        )

        numeric_columns = [
            "usage_count",
            "usage_duration_secs",
            "error_count"
        ]

        for column in numeric_columns:

            usage[column] = pd.to_numeric(
                usage[column],
                errors="coerce"
            )

        usage = (
            usage
            .dropna(
                subset=[
                    "usage_id",
                    "subscription_id",
                    "usage_date"
                ]
            )
            .drop_duplicates(
                subset=["usage_id"],
                keep="last"
            )
        )

        return usage


    #load_support_tickets data from db

    def load_support_tickets(self):

        query = """
        SELECT
            ticket_id,
            account_id,
            submitted_at,
            closed_at,
            resolution_time_hours,
            priority,
            first_response_time_minutes,
            satisfaction_score
        FROM raw_support_tickets;
        """

        support = pd.read_sql(
            query,
            self.engine
        )

        support["submitted_at"] = pd.to_datetime(
            support["submitted_at"],
            errors="coerce"
        )

        support["closed_at"] = pd.to_datetime(
            support["closed_at"],
            errors="coerce"
        )

        numeric_columns = [
            "resolution_time_hours",
            "first_response_time_minutes",
            "satisfaction_score"
        ]

        for column in numeric_columns:

            support[column] = pd.to_numeric(
                support[column],
                errors="coerce"
            )

        support = (
            support
            .dropna(
                subset=[
                    "ticket_id",
                    "account_id",
                    "submitted_at"
                ]
            )
            .drop_duplicates(
                subset=["ticket_id"],
                keep="last"
            )
        )

        return support


    #add_account features to the snapshots df

    def add_account_features(
        self,
        snapshots_df
    ):

        df = snapshots_df.copy()

        accounts = self.load_accounts()

        df = df.merge(
            accounts,
            on="account_id",
            how="left",
            validate="many_to_one"
        )

        df["tenure_days"] = (
            df["snapshot_date"]
            - df["signup_date"]
        ).dt.days


        return df


    #reactivation_count

    def count_reactivations(
        self,
        historical_subscriptions
    ):

        if historical_subscriptions.empty:

            return 0

        subscriptions = (
            historical_subscriptions
            .sort_values("start_date")
            .copy()
        )

        reactivation_count = 0

        for subscription in (
            subscriptions.itertuples(
                index=False
            )
        ):

            if (
                subscription.churn_flag == 1
                and
                pd.notna(subscription.end_date)
            ):

                later_subscription_exists = (
                    subscriptions["start_date"]
                    > subscription.end_date
                ).any()

                if later_subscription_exists:

                    reactivation_count += 1

        return reactivation_count

    def add_subscription_features(
            self,
            snapshots_df
    ):

        df = snapshots_df.copy()

        subscriptions = self.load_subscriptions()
        subscriptions_by_account = {
            account_id: group.copy()
            for account_id, group
            in subscriptions.groupby("account_id")
        }

        empty_subscriptions = (
            subscriptions.iloc[0:0].copy()
        )

        feature_rows = []
        for row in df[
            [
                "account_id",
                "snapshot_date"
            ]
        ].itertuples(index=False):

            account_id = row.account_id
            snapshot_date = row.snapshot_date
            account_subscriptions = (
                subscriptions_by_account.get(
                    account_id,
                    empty_subscriptions
                )
            )

            historical = (
                account_subscriptions[
                    account_subscriptions[
                        "start_date"
                    ] <= snapshot_date
                    ]
                .sort_values("start_date")
                .copy()
            )
            active = (
                historical[
                    historical["end_date"].isna()
                    |
                    (
                            historical["end_date"]
                            > snapshot_date
                    )
                    ]
                .sort_values("start_date")
                .copy()
            )
            completed = (
                historical[
                    historical["end_date"].notna()
                    &
                    (
                            historical["end_date"]
                            <= snapshot_date
                    )
                    ]
                .sort_values("end_date")
                .copy()
            )

            if not active.empty:

                current = active.iloc[-1]

                current_plan = (
                    current["plan_tier"]
                )

                current_seats = (
                    current["seats"]
                )

                current_mrr = (
                    current["mrr_amount"]
                )

                current_is_trial = (
                    current["is_trial"]
                )

                current_billing_frequency = (
                    current["billing_frequency"]
                )

                current_auto_renew = (
                    current["auto_renew_flag"]
                )

                subscription_age_days = (
                        snapshot_date
                        - current["start_date"]
                ).days

            else:

                current_plan = None
                current_seats = np.nan
                current_mrr = np.nan
                current_is_trial = np.nan
                current_billing_frequency = None
                current_auto_renew = np.nan
                subscription_age_days = np.nan

            completed_count = len(
                completed
            )

            historical_churn_count = (
                self.safe_sum(
                    completed["churn_flag"]
                )
            )

            reactivation_count = (
                self.count_reactivations(
                    historical
                )
            )

            feature_rows.append({

                "account_id":
                    account_id,

                "snapshot_date":
                    snapshot_date,

                # Current subscription
                "current_plan_tier":
                    current_plan,

                "current_seats":
                    current_seats,

                "current_mrr":
                    current_mrr,

                "current_is_trial":
                    current_is_trial,

                "current_billing_frequency":
                    current_billing_frequency,

                "current_auto_renew_flag":
                    current_auto_renew,

                "current_subscription_age_days":
                    subscription_age_days,

                # Subscription history
                "completed_subscription_count":
                    completed_count,

                "subscription_churn_count_to_date":
                    historical_churn_count,

                "reactivation_count_to_date":
                    reactivation_count,

            })

        subscription_features = pd.DataFrame(
            feature_rows
        )

        df = df.merge(
            subscription_features,
            on=[
                "account_id",
                "snapshot_date"
            ],
            how="left",
            validate="one_to_one"
        )

        return df


    #usage windows

    def get_usage_window(
        self,
        account_usage,
        snapshot_date,
        window_days
    ):

        window_start = (
            snapshot_date
            - pd.Timedelta(
                days=window_days
            )
        )

        return account_usage[
            account_usage["usage_date"]
            > window_start
        ].copy()


    # usage features

    def add_usage_features(
            self,
            snapshots_df
    ):

        df = snapshots_df.copy()



        usage = self.load_usage()
        subscriptions = self.load_subscriptions()

        subscription_accounts = (
            subscriptions[
                [
                    "subscription_id",
                    "account_id"
                ]
            ]
            .drop_duplicates(
                subset=["subscription_id"]
            )
        )

        usage = usage.merge(
            subscription_accounts,
            on="subscription_id",
            how="left",
            validate="many_to_one"
        )


        usage = usage.dropna(
            subset=["account_id"]
        )

        usage_by_account = {
            account_id: group.copy()
            for account_id, group
            in usage.groupby("account_id")
        }

        empty_usage = usage.iloc[0:0].copy()

        feature_rows = []

        for row in df[
            [
                "account_id",
                "snapshot_date",
                "current_seats"
            ]
        ].itertuples(index=False):

            account_id = row.account_id
            snapshot_date = row.snapshot_date
            current_seats = row.current_seats

            all_account_usage = (
                usage_by_account.get(
                    account_id,
                    empty_usage
                )
            )

            account_usage = all_account_usage[
                all_account_usage["usage_date"]
                <= snapshot_date
                ].copy()

            usage_7d = self.get_usage_window(
                account_usage,
                snapshot_date,
                7
            )

            usage_30d = self.get_usage_window(
                account_usage,
                snapshot_date,
                30
            )


            usage_count_7d = self.safe_sum(
                usage_7d["usage_count"]
            )

            usage_count_30d = self.safe_sum(
                usage_30d["usage_count"]
            )

            usage_duration_30d = self.safe_sum(
                usage_30d[
                    "usage_duration_secs"
                ]
            )

            unique_features_30d = (
                usage_30d[
                    "feature_name"
                ].nunique()
            )

            active_usage_days_30d = (
                usage_30d[
                    "usage_date"
                ]
                .dt.normalize()
                .nunique()
            )


            previous_7_start = (
                    snapshot_date
                    - pd.Timedelta(days=14)
            )

            recent_7_start = (
                    snapshot_date
                    - pd.Timedelta(days=7)
            )

            previous_7 = account_usage[
                (
                        account_usage["usage_date"]
                        > previous_7_start
                )
                &
                (
                        account_usage["usage_date"]
                        <= recent_7_start
                )
                ]

            previous_usage_7d = self.safe_sum(
                previous_7["usage_count"]
            )

            previous_30_start = (
                    snapshot_date
                    - pd.Timedelta(days=60)
            )

            recent_30_start = (
                    snapshot_date
                    - pd.Timedelta(days=30)
            )

            previous_30 = account_usage[
                (
                        account_usage["usage_date"]
                        > previous_30_start
                )
                &
                (
                        account_usage["usage_date"]
                        <= recent_30_start
                )
                ]

            previous_usage_30d = self.safe_sum(
                previous_30["usage_count"]
            )

            if not account_usage.empty:

                days_since_last_usage = (
                        snapshot_date
                        - account_usage[
                            "usage_date"
                        ].max()
                ).days

            else:

                days_since_last_usage = np.nan

            usage_log_change_7d = (
                self.log_change(
                    current_value=usage_count_7d,
                    previous_value=previous_usage_7d
                )
            )

            usage_log_change_30d = (
                self.log_change(
                    current_value=usage_count_30d,
                    previous_value=previous_usage_30d
                )
            )

            errors_30d = self.safe_sum(
                usage_30d["error_count"]
            )

            error_rate_30d = (
                errors_30d / usage_count_30d
                if usage_count_30d > 0
                else 0
            )

            usage_per_seat_30d = (
                usage_count_30d / current_seats
                if (
                        pd.notna(current_seats)
                        and current_seats > 0
                )
                else np.nan
            )


            feature_rows.append({

                "account_id":
                    account_id,

                "snapshot_date":
                    snapshot_date,

                "usage_count_7d":
                    usage_count_7d,

                "usage_count_30d":
                    usage_count_30d,

                "usage_duration_secs_30d":
                    usage_duration_30d,

                "unique_features_30d":
                    unique_features_30d,

                "active_usage_days_30d":
                    active_usage_days_30d,

                "days_since_last_usage":
                    days_since_last_usage,

                "usage_log_change_7d":
                    usage_log_change_7d,

                "usage_log_change_30d":
                    usage_log_change_30d,

                "error_rate_30d":
                    error_rate_30d,

                "usage_per_seat_30d":
                    usage_per_seat_30d
            })



        usage_features = pd.DataFrame(
            feature_rows
        )

        df = df.merge(
            usage_features,
            on=[
                "account_id",
                "snapshot_date"
            ],
            how="left",
            validate="one_to_one"
        )

        return df
    #support features

    def add_support_features(
            self,
            snapshots_df
    ):

        df = snapshots_df.copy()

        support = self.load_support_tickets()

        support_by_account = {
            account_id: group.copy()
            for account_id, group
            in support.groupby("account_id")
        }
        empty_support = support.iloc[0:0].copy()

        feature_rows = []

        for row in df[
            [
                "account_id",
                "snapshot_date"
            ]
        ].itertuples(index=False):

            account_id = row.account_id
            snapshot_date = row.snapshot_date
            all_account_support = (
                support_by_account.get(
                    account_id,
                    empty_support
                )
            )

            account_support = all_account_support[
                all_account_support["submitted_at"]
                <= snapshot_date
                ].copy()

            open_tickets = account_support[
                account_support["closed_at"].isna()
                |
                (
                        account_support["closed_at"]
                        > snapshot_date
                )
                ].copy()

            recent_30_start = (
                    snapshot_date
                    - pd.Timedelta(days=30)
            )

            recent_90_start = (
                    snapshot_date
                    - pd.Timedelta(days=90)
            )

            previous_30_start = (
                    snapshot_date
                    - pd.Timedelta(days=60)
            )

            tickets_30d = account_support[
                account_support["submitted_at"]
                > recent_30_start
                ].copy()

            tickets_90d = account_support[
                account_support["submitted_at"]
                > recent_90_start
                ].copy()
            previous_tickets_30d = account_support[
                (
                        account_support["submitted_at"]
                        > previous_30_start
                )
                &
                (
                        account_support["submitted_at"]
                        <= recent_30_start
                )
                ].copy()

            closed_tickets_90d = tickets_90d[
                tickets_90d["closed_at"].notna()
                &
                (
                        tickets_90d["closed_at"]
                        <= snapshot_date
                )
                ].copy()

            priority_values = (
                tickets_30d["priority"]
                .astype("string")
                .str.strip()
                .str.lower()
            )

            high_priority_count_30d = (
                priority_values
                .isin(
                    [
                        "high",
                        "urgent",
                        "critical"
                    ]
                )
                .sum()
            )

            ticket_count_30d = len(
                tickets_30d
            )

            # Needed internally for ticket_log_change_30d.
            # It will NOT be included as a final model feature.
            previous_ticket_count_30d = len(
                previous_tickets_30d
            )

            if not account_support.empty:

                days_since_last_ticket = (
                        snapshot_date
                        - account_support[
                            "submitted_at"
                        ].max()
                ).days

            else:

                days_since_last_ticket = np.nan

            average_satisfaction_90d = (
                closed_tickets_90d[
                    "satisfaction_score"
                ].mean()
            )

            has_satisfaction_90d = int(
                closed_tickets_90d[
                    "satisfaction_score"
                ]
                .notna()
                .any()
            )

            ticket_log_change_30d = (
                self.log_change(
                    current_value=ticket_count_30d,
                    previous_value=previous_ticket_count_30d
                )
            )

            feature_rows.append({

                "account_id":
                    account_id,

                "snapshot_date":
                    snapshot_date,


                "ticket_count_30d":
                    ticket_count_30d,


                "open_ticket_count":
                    len(open_tickets),


                "high_priority_ticket_count_30d":
                    high_priority_count_30d,


                "avg_satisfaction_90d":
                    average_satisfaction_90d,

                "has_satisfaction_90d":
                    has_satisfaction_90d,


                "days_since_last_ticket":
                    days_since_last_ticket,


                "ticket_log_change_30d":
                    ticket_log_change_30d
            })

        support_features = pd.DataFrame(
            feature_rows
        )

        df = df.merge(
            support_features,
            on=[
                "account_id",
                "snapshot_date"
            ],
            how="left",
            validate="one_to_one"
        )

        return df

    #building all features
    def build_all_features(
            self,
            snapshots_df,
            target_column=None
    ):

        required_columns = [
            "account_id",
            "snapshot_date"
        ]
        if target_column is not None:
            required_columns.append(
                target_column
            )

        missing_columns = [
            column
            for column in required_columns
            if column not in snapshots_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: "
                f"{missing_columns}"
            )

        df = snapshots_df.copy()

        df["snapshot_date"] = pd.to_datetime(
            df["snapshot_date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "account_id",
                "snapshot_date"
            ]
        )

        initial_row_count = len(df)

        if target_column is not None:

            initial_positive_count = int(
                df[target_column].sum()
            )

        else:

            initial_positive_count = None

        duplicate_count = (
            df.duplicated(
                subset=[
                    "account_id",
                    "snapshot_date"
                ]
            ).sum()
        )

        if duplicate_count > 0:
            raise ValueError(
                f"Snapshots contain {duplicate_count} "
                f"duplicate customer-snapshot rows."
            )

        df = self.add_account_features(
            df
        )

        df = self.add_subscription_features(
            df
        )

        df = self.add_usage_features(
            df
        )

        df = self.add_support_features(
            df
        )

        if len(df) != initial_row_count:
            raise ValueError(
                "Feature engineering changed the row count. "
                f"Before: {initial_row_count}; "
                f"after: {len(df)}."
            )

        final_duplicate_count = (
            df.duplicated(
                subset=[
                    "account_id",
                    "snapshot_date"
                ]
            ).sum()
        )

        if final_duplicate_count > 0:
            raise ValueError(
                "Feature engineering created duplicate "
                "customer-snapshot rows."
            )
        if target_column is not None:

            final_positive_count = int(
                df[target_column].sum()
            )

            if final_positive_count != initial_positive_count:
                raise ValueError(
                    "Feature engineering changed the number "
                    f"of positive rows in {target_column}. "
                    f"Before: {initial_positive_count}; "
                    f"after: {final_positive_count}."
                )
        df = (
            df
            .sort_values(
                [
                    "snapshot_date",
                    "account_id"
                ]
            )
            .reset_index(drop=True)
        )

        return df