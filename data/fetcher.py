"""Market data ingestion and DuckDB storage."""

import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path


ASSETS = ["ES=F", "NQ=F", "GC=F", "CL=F", "TLT", "SPY", "DX-Y.NYB", "^VIX", "ZN=F"]
ASSET_LABELS = {
    "ES=F": "ES", "NQ=F": "NQ", "GC=F": "GC", "CL=F": "CL",
    "TLT": "TLT", "SPY": "SPY", "DX-Y.NYB": "DXY", "^VIX": "VIX", "ZN=F": "ZN"
}
DB_PATH = Path(__file__).parent.parent / "aria.db"


class DataFetcher:
    """Pulls OHLCV data from yfinance and stores in DuckDB."""

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)
        self._init_db()

    def _init_db(self):
        """Create the OHLCV table if it doesn't exist."""
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                timestamp TIMESTAMP,
                asset VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (timestamp, asset)
            )
        """)
        con.close()
        logger.info("DuckDB initialized at {}", self.db_path)

    def _get_latest_timestamp(self, asset_label: str):
        """Get the most recent timestamp for an asset in the DB."""
        con = duckdb.connect(self.db_path)
        result = con.execute(
            "SELECT MAX(timestamp) FROM ohlcv WHERE asset = ?", [asset_label]
        ).fetchone()
        con.close()
        return result[0] if result and result[0] else None

    def fetch_asset(self, ticker: str, period: str = "2y", interval: str = "1h"):
        """Fetch OHLCV data for a single asset and upsert into DuckDB."""
        label = ASSET_LABELS.get(ticker, ticker)
        logger.info("Fetching {} ({})", ticker, label)

        try:
            # yfinance intraday limits: 1m=7d, 5m=60d, 1h=730d, 1d=unlimited
            # Auto-select best interval for the requested period
            period_days = {"1d": 1, "5d": 5, "7d": 7, "1mo": 30, "3mo": 90,
                           "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "max": 9999}
            days = period_days.get(period, 730)

            if days <= 7:
                effective_interval = "5m"
            elif days <= 60:
                effective_interval = "5m"
            elif days <= 730:
                effective_interval = "1h"
            else:
                effective_interval = "1d"

            # Override if caller explicitly set a valid interval
            if interval in ("1d", "1h", "5m") and interval != "1m":
                effective_interval = interval

            data = yf.download(
                ticker, period=period, interval=effective_interval,
                progress=False, auto_adjust=True
            )

            if data.empty:
                logger.warning("No data returned for {}", ticker)
                return 0

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.reset_index()
            date_col = "Datetime" if "Datetime" in data.columns else "Date"
            data = data.rename(columns={
                date_col: "timestamp", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume"
            })
            data["asset"] = label
            data["volume"] = data["volume"].fillna(0).astype(int)
            data = data[["timestamp", "asset", "open", "high", "low", "close", "volume"]]
            data = data.dropna(subset=["open", "high", "low", "close"])

            con = duckdb.connect(self.db_path)
            con.execute("BEGIN TRANSACTION")
            # Upsert: delete existing then insert
            min_ts = data["timestamp"].min()
            max_ts = data["timestamp"].max()
            con.execute(
                "DELETE FROM ohlcv WHERE asset = ? AND timestamp >= ? AND timestamp <= ?",
                [label, min_ts, max_ts]
            )
            con.execute("INSERT INTO ohlcv SELECT * FROM data")
            con.execute("COMMIT")
            count = len(data)
            con.close()
            logger.info("Stored {} bars for {}", count, label)
            return count

        except Exception as e:
            logger.error("Failed to fetch {}: {}", ticker, e)
            return 0

    def fetch_all(self, period: str = "2y"):
        """Fetch data for all tracked assets."""
        total = 0
        for ticker in ASSETS:
            total += self.fetch_asset(ticker, period=period)
        logger.info("Total bars fetched: {}", total)
        return total

    def refresh(self):
        """Incremental refresh — fetch last 7 days of data."""
        total = 0
        for ticker in ASSETS:
            total += self.fetch_asset(ticker, period="7d", interval="5m")
        logger.info("Refresh complete: {} bars updated", total)
        return total

    def fill_gaps(self):
        """Forward-fill NaN values in OHLCV data without reindexing."""
        con = duckdb.connect(self.db_path)
        for label in ASSET_LABELS.values():
            df = con.execute(
                "SELECT * FROM ohlcv WHERE asset = ? ORDER BY timestamp", [label]
            ).fetchdf()

            if df.empty:
                continue

            # Just forward-fill NaN prices, don't reindex (preserves natural timestamps)
            before = df[["open", "high", "low", "close"]].isna().sum().sum()
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
            df["volume"] = df["volume"].fillna(0).astype(int)
            df = df.dropna(subset=["open"])
            after = before - df[["open", "high", "low", "close"]].isna().sum().sum()

            if after > 0:
                con.execute("DELETE FROM ohlcv WHERE asset = ?", [label])
                con.execute("INSERT INTO ohlcv SELECT * FROM df")
                logger.info("Filled {} NaN values for {}", after, label)

        con.close()
        logger.info("Gap filling complete")

    def get_data(self, asset: str, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """Retrieve OHLCV data for an asset from DuckDB."""
        con = duckdb.connect(self.db_path)
        query = "SELECT * FROM ohlcv WHERE asset = ?"
        params = [asset]
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)
        query += " ORDER BY timestamp"
        df = con.execute(query, params).fetchdf()
        con.close()
        return df

    def get_all_assets(self):
        """Return list of all assets in the database."""
        con = duckdb.connect(self.db_path)
        result = con.execute("SELECT DISTINCT asset FROM ohlcv").fetchall()
        con.close()
        return [r[0] for r in result]

    def get_coverage(self) -> dict:
        """Return data coverage stats per asset."""
        con = duckdb.connect(self.db_path)
        result = con.execute("""
            SELECT asset, COUNT(*) as bars, MIN(timestamp) as start, MAX(timestamp) as end
            FROM ohlcv GROUP BY asset ORDER BY asset
        """).fetchdf()
        con.close()
        return result.to_dict("records")
