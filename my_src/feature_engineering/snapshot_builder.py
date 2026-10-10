import pandas as pd


class SnapshotBuilder:

    def __init__(
        self,
        engine,
        observation_end_date,
        prediction_horizon_days=7
    ):
        self.engine = engine

        self.observation_end_date = pd.to_datetime(
            observation_end_date
        )

        self.prediction_horizon_days = (
            prediction_horizon_days
        )

    #load accounts
    def load_accounts(self):

        query = "SELECT account_id, signup_date FROM raw_accounts;"


        accounts = pd.read_sql(
            query,
            self.engine
        )

        accounts["signup_date"] = pd.to_datetime(
            accounts["signup_date"],
            errors="coerce"
        )

        return accounts

    def load_subscriptions(self):

        query = "SELECT subscription_id, account_id, start_date, end_date, churn_flag FROM raw_subscriptions;"

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

        return subscriptions


    def get_latest_valid_snapshot_date(self):

        latest_valid_snapshot = (
            self.observation_end_date
            - pd.Timedelta(
                days=self.prediction_horizon_days
            )
        )

        return latest_valid_snapshot


    def create_snapshot_dates(
        self,
        start_date,
        end_date=None,
        frequency="7D"
    ):

        start_date = pd.to_datetime(
            start_date
        )

        latest_valid_snapshot = (
            self.get_latest_valid_snapshot_date()
        )

        if end_date is None:

            final_end_date = (
                latest_valid_snapshot
            )

        else:

            requested_end_date = pd.to_datetime(
                end_date
            )

            final_end_date = min(
                requested_end_date,
                latest_valid_snapshot
            )

        if start_date > final_end_date:

            raise ValueError(
                "start_date is later than the latest "
                "valid snapshot date."
            )

        snapshot_dates = pd.date_range(
            start=start_date,
            end=final_end_date,
            freq=frequency
        )

        return snapshot_dates

    def get_active_accounts(
        self,
        subscriptions,
        snapshot_date
    ):


        active_subscriptions = subscriptions[
            (
                subscriptions["start_date"]
                <= snapshot_date
            )
            &
            (
                subscriptions["end_date"].isna()
                |
                (
                    subscriptions["end_date"]
                    > snapshot_date
                )
            )
        ].copy()

        active_accounts = (
            active_subscriptions[
                ["account_id"]
            ]
            .drop_duplicates()
        )

        return active_accounts

    def build_snapshots(
        self,
        start_date,
        end_date=None,
        frequency="7D"
    ):


        accounts = self.load_accounts()

        subscriptions = self.load_subscriptions()

        snapshot_dates = self.create_snapshot_dates(
            start_date=start_date,
            end_date=end_date,
            frequency=frequency
        )

        snapshots = []

        for snapshot_date in snapshot_dates:

            active_accounts = (
                self.get_active_accounts(
                    subscriptions=subscriptions,
                    snapshot_date=snapshot_date
                )
            )

            # Attach signup information
            active_accounts = (
                active_accounts.merge(
                    accounts,
                    on="account_id",
                    how="left",
                    validate="one_to_one"
                )
            )

            # Customer cannot appear before signup
            active_accounts = active_accounts[
                active_accounts["signup_date"]
                <= snapshot_date
            ].copy()

            active_accounts[
                "snapshot_date"
            ] = snapshot_date

            snapshots.append(
                active_accounts[
                    [
                        "account_id",
                        "snapshot_date"
                    ]
                ]
            )

        if not snapshots:

            return pd.DataFrame(
                columns=[
                    "account_id",
                    "snapshot_date"
                ]
            )

        snapshots_df = pd.concat(
            snapshots,
            ignore_index=True
        )

        # Final duplicate protection
        snapshots_df = (
            snapshots_df
            .drop_duplicates(
                subset=[
                    "account_id",
                    "snapshot_date"
                ]
            )
        )

        snapshots_df = (
            snapshots_df
            .sort_values(
                [
                    "snapshot_date",
                    "account_id"
                ]
            )
            .reset_index(drop=True)
        )

        return snapshots_df