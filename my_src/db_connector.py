
import os
import ssl
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL


class DatabaseConnector:

    def __init__(self, database=None):

        self.project_root = Path(__file__).resolve().parents[1]

        load_dotenv(self.project_root / ".env")

        self.user = os.getenv("MYSQL_USER")
        self.password = os.getenv("MYSQL_PASSWORD")
        self.host = os.getenv("MYSQL_HOST")
        self.port = os.getenv("MYSQL_PORT", "3306")
        self.database = database or os.getenv("MYSQL_DATABASE")

        self.ssl_ca = os.getenv("MYSQL_SSL_CA")

        self.engine = self._create_engine()


    def _create_engine(self) -> Engine:

        required = {
            "MYSQL_USER": self.user,
            "MYSQL_PASSWORD": self.password,
            "MYSQL_HOST": self.host,
            "MYSQL_DATABASE": self.database,
        }

        missing = [
            name
            for name, value in required.items()
            if not value
        ]

        if missing:
            raise ValueError(
                f"Missing database settings: {', '.join(missing)}"
            )

        connection_url = URL.create(
            drivername="mysql+pymysql",
            username=self.user,
            password=self.password,
            host=self.host,
            port=int(self.port),
            database=self.database,
        )

        connect_args = {}

        if self.ssl_ca:

            ca_path = Path(self.ssl_ca).expanduser()

            if not ca_path.is_absolute():
                ca_path = self.project_root / ca_path

            if not ca_path.is_file():
                raise FileNotFoundError(
                    f"CA certificate not found: {ca_path}"
                )

            ssl_context = ssl.create_default_context(
                cafile=str(ca_path)
            )

            ssl_context.verify_mode = ssl.CERT_REQUIRED
            ssl_context.check_hostname = True

            connect_args["ssl"] = ssl_context

        elif self.host not in ("localhost", "127.0.0.1"):

            raise ValueError(
                "MYSQL_SSL_CA is required for remote MySQL connections."
            )

        # -------------------------------------------------
        # CREATE ENGINE
        # -------------------------------------------------

        engine = create_engine(
            connection_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_recycle=1800,
        )

        return engine


    def get_engine(self):

        return self.engine


class MySQLTableExtractor:

    def __init__(self, engine):

        self.engine = engine


    def list_tables(self):

        query = "SHOW TABLES;"

        tables = pd.read_sql(
            query,
            self.engine
        )

        return tables.iloc[:, 0].tolist()


    def load_table(self, table_name):

        if table_name not in self.list_tables():
            raise ValueError(
                f"Unknown table: {table_name}"
            )

        query = f"SELECT * FROM `{table_name}`;"

        df = pd.read_sql(
            query,
            self.engine
        )

        return df


    def extract_all_tables(self, output_folder):

        output_folder = Path(output_folder)

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        tables = self.list_tables()

        for table in tables:

            df = self.load_table(table)

            file_path = output_folder / f"{table}.csv"

            df.to_csv(
                file_path,
                index=False
            )

            print(
                f"Saved {table} -> {file_path}"
            )


class MySQLTableLoader:

    def __init__(self, engine):

        self.engine = engine


    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace"
    ):

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a pandas DataFrame"
            )

        if if_exists not in [
            "fail",
            "replace",
            "append"
        ]:
            raise ValueError(
                "if_exists must be: fail, replace, or append"
            )

        df.to_sql(
            name=table_name,
            con=self.engine,
            if_exists=if_exists,
            index=False,
            chunksize=5000
        )

        print(
            f"Loaded {len(df):,} rows "
            f"into {table_name}"
        )


    def load_cleaned_csv(
        self,
        file_path,
        table_name=None,
        if_exists="replace"
    ):

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        df = pd.read_csv(file_path)

        if table_name is None:

            table_name = file_path.stem

            if table_name.endswith("_clean"):
                table_name = table_name.removesuffix(
                    "_clean"
                )

        self.load_dataframe(
            df=df,
            table_name=table_name,
            if_exists=if_exists
        )


    def load_all_cleaned_csvs(
        self,
        input_folder,
        if_exists="replace"
    ):

        input_folder = Path(input_folder)

        if not input_folder.exists():
            raise FileNotFoundError(
                f"Folder not found: {input_folder}"
            )

        csv_files = sorted(
            input_folder.glob("*.csv")
        )

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {input_folder}"
            )

        for file_path in csv_files:

            table_name = file_path.stem

            if table_name.endswith("_clean"):
                table_name = table_name.removesuffix(
                    "_clean"
                )

            self.load_cleaned_csv(
                file_path=file_path,
                table_name=table_name,
                if_exists=if_exists
            )
