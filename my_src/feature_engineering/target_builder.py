import pandas as pd


class TargetBuilder:

    def __init__(self, engine):
        self.engine = engine


    def load_churn_events(self):


        query = "SELECT account_id, churn_date FROM raw_churn_events;"

        churn_events = pd.read_sql(
            query,
            self.engine
        )

        churn_events["churn_date"] = pd.to_datetime(
            churn_events["churn_date"],
            errors="coerce"
        )

        # Remove events where churn_date could not be interpreted.
        churn_events = churn_events.dropna(
            subset=[
                "account_id",
                "churn_date"
            ]
        )

        return churn_events

    def add_churn_target(
            self,
            snapshots_df,
            prediction_window_days=7
    ):
        df = snapshots_df.copy()

        # Ensure snapshot_date is datetime
        df["snapshot_date"] = pd.to_datetime(
            df["snapshot_date"],
            errors="coerce"
        )

        churn_events = self.load_churn_events()

        churn_lookup = (
            churn_events
            .groupby("account_id")["churn_date"]
            .apply(list)
            .to_dict()
        )

        targets = []

        # Check every customer at every snapshot
        for row in df.itertuples(index=False):
            account_id = row.account_id
            snapshot_date = row.snapshot_date

            prediction_end = (
                    snapshot_date
                    + pd.Timedelta(
                days=prediction_window_days
            )
            )

            account_churn_dates = churn_lookup.get(
                account_id,
                []
            )

            churn_in_window = any(
                (
                        churn_date > snapshot_date
                        and
                        churn_date <= prediction_end
                )
                for churn_date in account_churn_dates
            )

            targets.append(
                int(churn_in_window)
            )

        # Dynamic target name
        target_column = (
            f"churn_next_{prediction_window_days}d"
        )

        df[target_column] = targets

        return df