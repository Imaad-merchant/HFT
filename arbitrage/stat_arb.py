"""Statistical arbitrage engine — pair discovery, spread monitoring, health tracking."""

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from pathlib import Path
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

DB_PATH = Path(__file__).parent.parent / "aria.db"

ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 3.5
COINT_PVALUE_THRESHOLD = 0.05
COINT_RETIRE_THRESHOLD = 0.10
RETEST_INTERVAL_DAYS = 5


class StatArbEngine:
    """Discovers and monitors cointegrated pairs for statistical arbitrage."""

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)
        self._init_db()
        self.live_pairs = []

    def _init_db(self):
        """Create stat arb tables."""
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS pairs (
                asset1 VARCHAR,
                asset2 VARCHAR,
                hedge_ratio DOUBLE,
                half_life DOUBLE,
                adf_pvalue DOUBLE,
                coint_pvalue DOUBLE,
                status VARCHAR DEFAULT 'ACTIVE',
                last_tested TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (asset1, asset2)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS pair_signals (
                timestamp TIMESTAMP,
                asset1 VARCHAR,
                asset2 VARCHAR,
                spread_z DOUBLE,
                signal VARCHAR,
                hedge_ratio DOUBLE
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS pair_status (
                timestamp TIMESTAMP,
                asset1 VARCHAR,
                asset2 VARCHAR,
                status VARCHAR,
                coint_pvalue DOUBLE,
                reason VARCHAR
            )
        """)
        con.close()

    def _compute_half_life(self, spread: pd.Series) -> float:
        """Estimate half-life of mean reversion via OLS on lagged spread."""
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        common_idx = spread_lag.index.intersection(spread_diff.index)
        spread_lag = spread_lag.loc[common_idx]
        spread_diff = spread_diff.loc[common_idx]

        if len(spread_lag) < 20:
            return np.inf

        X = add_constant(spread_lag.values)
        y = spread_diff.values
        try:
            result = OLS(y, X).fit()
            theta = result.params[1]
            if theta >= 0:
                return np.inf
            return -np.log(2) / theta
        except Exception:
            return np.inf

    def discover_pairs(self):
        """Test all asset pairs for cointegration and store results."""
        con = duckdb.connect(self.db_path)
        assets = [r[0] for r in con.execute("SELECT DISTINCT asset FROM ohlcv").fetchall()]

        # Get close prices for all assets
        closes = {}
        for asset in assets:
            df = con.execute(
                "SELECT timestamp, close FROM ohlcv WHERE asset = ? ORDER BY timestamp", [asset]
            ).fetchdf()
            if not df.empty:
                closes[asset] = df.set_index("timestamp")["close"]
        con.close()

        if len(closes) < 2:
            logger.warning("Not enough assets for pair discovery")
            return

        # Align all series
        close_df = pd.DataFrame(closes).dropna()
        if len(close_df) < 100:
            logger.warning("Not enough aligned data for pair discovery")
            return

        found_pairs = []
        for a1, a2 in combinations(close_df.columns, 2):
            try:
                # Engle-Granger cointegration test
                score, pvalue, _ = coint(close_df[a1].values, close_df[a2].values)

                if pvalue < COINT_PVALUE_THRESHOLD:
                    # Compute hedge ratio
                    X = add_constant(close_df[a2].values)
                    result = OLS(close_df[a1].values, X).fit()
                    hedge_ratio = result.params[1]

                    # Compute spread and its properties
                    spread = close_df[a1] - hedge_ratio * close_df[a2]
                    half_life = self._compute_half_life(spread)

                    # ADF test on spread
                    adf_result = adfuller(spread.dropna().values)
                    adf_pvalue = adf_result[1]

                    if half_life < 100 and adf_pvalue < 0.05:
                        found_pairs.append({
                            "asset1": a1, "asset2": a2,
                            "hedge_ratio": hedge_ratio,
                            "half_life": half_life,
                            "adf_pvalue": adf_pvalue,
                            "coint_pvalue": pvalue,
                            "status": "ACTIVE",
                            "last_tested": pd.Timestamp.now(),
                        })
                        logger.info("Cointegrated pair: {}-{} (p={:.4f}, hl={:.1f})", a1, a2, pvalue, half_life)

            except Exception as e:
                logger.debug("Coint test failed for {}-{}: {}", a1, a2, e)

        # Store pairs
        if found_pairs:
            con = duckdb.connect(self.db_path)
            con.execute("BEGIN TRANSACTION")
            for pair in found_pairs:
                con.execute("DELETE FROM pairs WHERE asset1 = ? AND asset2 = ?",
                            [pair["asset1"], pair["asset2"]])
                df = pd.DataFrame([pair])
                con.execute("INSERT INTO pairs SELECT * FROM df")
            con.execute("COMMIT")
            con.close()
            logger.info("Discovered {} cointegrated pairs", len(found_pairs))
        else:
            logger.info("No cointegrated pairs found")

    def get_active_pairs(self) -> list[dict]:
        """Get all active pairs from DB."""
        con = duckdb.connect(self.db_path)
        df = con.execute("SELECT * FROM pairs WHERE status = 'ACTIVE'").fetchdf()
        con.close()
        return df.to_dict("records")

    def compute_spread_z(self, asset1: str, asset2: str, hedge_ratio: float, lookback: int = 200) -> float | None:
        """Compute current normalized spread z-score for a pair."""
        con = duckdb.connect(self.db_path)
        df1 = con.execute(
            "SELECT timestamp, close FROM ohlcv WHERE asset = ? ORDER BY timestamp DESC LIMIT ?",
            [asset1, lookback]
        ).fetchdf().sort_values("timestamp")
        df2 = con.execute(
            "SELECT timestamp, close FROM ohlcv WHERE asset = ? ORDER BY timestamp DESC LIMIT ?",
            [asset2, lookback]
        ).fetchdf().sort_values("timestamp")
        con.close()

        if df1.empty or df2.empty:
            return None

        merged = df1.merge(df2, on="timestamp", suffixes=("_1", "_2"))
        if len(merged) < 20:
            return None

        spread = merged["close_1"] - hedge_ratio * merged["close_2"]
        z = (spread.iloc[-1] - spread.mean()) / (spread.std() + 1e-10)
        return float(z)

    def generate_signals(self) -> list[dict]:
        """Generate entry/exit/stop signals for all active pairs."""
        pairs = self.get_active_pairs()
        signals = []

        for pair in pairs:
            z = self.compute_spread_z(pair["asset1"], pair["asset2"], pair["hedge_ratio"])
            if z is None:
                continue

            signal = "HOLD"
            if abs(z) > STOP_Z:
                signal = "STOP"
            elif abs(z) > ENTRY_Z:
                signal = "ENTRY_SHORT" if z > 0 else "ENTRY_LONG"
            elif abs(z) < EXIT_Z:
                signal = "EXIT"

            sig = {
                "timestamp": pd.Timestamp.now(),
                "asset1": pair["asset1"],
                "asset2": pair["asset2"],
                "spread_z": z,
                "signal": signal,
                "hedge_ratio": pair["hedge_ratio"],
            }
            signals.append(sig)

            if signal != "HOLD":
                logger.info("StatArb signal: {}-{} z={:.2f} -> {}", pair["asset1"], pair["asset2"], z, signal)

        # Store signals
        if signals:
            con = duckdb.connect(self.db_path)
            df = pd.DataFrame(signals)
            con.execute("INSERT INTO pair_signals SELECT * FROM df")
            con.close()

        return signals

    def retest_pairs(self):
        """Re-test cointegration for active pairs and retire broken ones."""
        pairs = self.get_active_pairs()
        con = duckdb.connect(self.db_path)

        for pair in pairs:
            a1, a2 = pair["asset1"], pair["asset2"]
            df1 = con.execute(
                "SELECT timestamp, close FROM ohlcv WHERE asset = ? ORDER BY timestamp", [a1]
            ).fetchdf()
            df2 = con.execute(
                "SELECT timestamp, close FROM ohlcv WHERE asset = ? ORDER BY timestamp", [a2]
            ).fetchdf()

            if df1.empty or df2.empty:
                continue

            merged = df1.merge(df2, on="timestamp", suffixes=("_1", "_2"))
            if len(merged) < 100:
                continue

            try:
                _, pvalue, _ = coint(merged["close_1"].values, merged["close_2"].values)
                new_status = pair["status"]
                reason = "routine retest"

                if pvalue > COINT_RETIRE_THRESHOLD:
                    new_status = "RETIRING"
                    reason = f"cointegration degraded (p={pvalue:.4f})"
                    logger.warning("Retiring pair {}-{}: {}", a1, a2, reason)

                if pvalue > 0.20:
                    new_status = "RETIRED"
                    reason = f"cointegration broken (p={pvalue:.4f})"
                    logger.warning("Retired pair {}-{}: {}", a1, a2, reason)

                con.execute("""
                    UPDATE pairs SET coint_pvalue = ?, status = ?, last_tested = CURRENT_TIMESTAMP
                    WHERE asset1 = ? AND asset2 = ?
                """, [pvalue, new_status, a1, a2])

                # Log status change
                status_log = pd.DataFrame([{
                    "timestamp": pd.Timestamp.now(),
                    "asset1": a1, "asset2": a2,
                    "status": new_status, "coint_pvalue": pvalue,
                    "reason": reason,
                }])
                con.execute("INSERT INTO pair_status SELECT * FROM status_log")

            except Exception as e:
                logger.error("Retest failed for {}-{}: {}", a1, a2, e)

        con.close()

    def get_correlation_regime(self) -> dict:
        """Track rolling correlations and detect breakdowns."""
        con = duckdb.connect(self.db_path)
        assets = [r[0] for r in con.execute("SELECT DISTINCT asset FROM ohlcv").fetchall()]

        closes = {}
        for asset in assets:
            df = con.execute(
                "SELECT timestamp, close FROM ohlcv WHERE asset = ? ORDER BY timestamp DESC LIMIT 500",
                [asset]
            ).fetchdf().sort_values("timestamp")
            if not df.empty:
                closes[asset] = df.set_index("timestamp")["close"]
        con.close()

        close_df = pd.DataFrame(closes).dropna()
        if len(close_df) < 60:
            return {"breakdowns": [], "regime": "NORMAL"}

        returns = close_df.pct_change().dropna()
        breakdowns = []

        for a1, a2 in combinations(returns.columns, 2):
            corr_20 = returns[a1].rolling(20).corr(returns[a2])
            corr_200 = returns[a1].rolling(200).corr(returns[a2])
            corr_std = returns[a1].rolling(200).corr(returns[a2]).rolling(60).std()

            if corr_std.iloc[-1] > 1e-10:
                z = (corr_20.iloc[-1] - corr_200.iloc[-1]) / corr_std.iloc[-1]
                if abs(z) > 2.0:
                    breakdowns.append({
                        "pair": f"{a1}-{a2}",
                        "corr_20d": float(corr_20.iloc[-1]),
                        "corr_200d": float(corr_200.iloc[-1]),
                        "z_score": float(z),
                    })

        regime = "BREAKDOWN" if len(breakdowns) > 3 else "NORMAL"
        return {"breakdowns": breakdowns, "regime": regime}

    def get_all_spread_zscores(self) -> dict[str, float]:
        """Get spread z-scores for all active pairs."""
        pairs = self.get_active_pairs()
        z_scores = {}
        for pair in pairs:
            key = f"{pair['asset1']}_{pair['asset2']}"
            z = self.compute_spread_z(pair["asset1"], pair["asset2"], pair["hedge_ratio"])
            z_scores[key] = z if z is not None else 0.0
        return z_scores
