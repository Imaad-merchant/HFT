"""Topological Data Analysis feature extraction using giotto-tda."""

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from pathlib import Path

try:
    from gtda.time_series import SingleTakensEmbedding
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import (
        PersistenceEntropy,
        Amplitude,
        NumberOfPoints,
        BettiCurve,
    )
    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False
    logger.warning("giotto-tda not installed — TDA features will be zeros")

DB_PATH = Path(__file__).parent.parent / "aria.db"

WINDOW_SIZE = 60
TAKENS_DIM = 3
TAKENS_DELAY = 5
LIFETIME_THRESHOLD = 0.05


class TDAFeatureExtractor:
    """Extracts topological features from price series via persistent homology."""

    def __init__(self, db_path: str = None, window_size: int = WINDOW_SIZE):
        self.db_path = str(db_path or DB_PATH)
        self.window_size = window_size
        self._init_db()

        if GTDA_AVAILABLE:
            self.embedder = SingleTakensEmbedding(
                parameters_type="fixed",
                dimension=TAKENS_DIM,
                time_delay=TAKENS_DELAY
            )
            self.persistence = VietorisRipsPersistence(
                homology_dimensions=[0, 1],
                max_edge_length=np.inf,
                n_jobs=-1
            )
            self.entropy = PersistenceEntropy()
            self.amplitude = Amplitude(metric="landscape")
            self.n_points = NumberOfPoints()

    def _init_db(self):
        """Create TDA tables if they don't exist."""
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS tda_features (
                timestamp TIMESTAMP,
                asset VARCHAR,
                betti_h0 DOUBLE,
                betti_h1 DOUBLE,
                entropy_h0 DOUBLE,
                entropy_h1 DOUBLE,
                amplitude DOUBLE,
                n_significant_h0 INTEGER,
                n_significant_h1 INTEGER,
                complexity_score DOUBLE,
                PRIMARY KEY (timestamp, asset)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS tda_events (
                timestamp TIMESTAMP,
                asset VARCHAR,
                event_type VARCHAR,
                betti_h0_before DOUBLE,
                betti_h0_after DOUBLE,
                betti_h1_before DOUBLE,
                betti_h1_after DOUBLE,
                description VARCHAR
            )
        """)
        con.close()

    def _extract_window_features(self, prices: np.ndarray) -> dict:
        """Extract TDA features from a single price window."""
        if not GTDA_AVAILABLE or len(prices) < self.window_size:
            return self._zero_features()

        try:
            # Normalize prices to [0, 1]
            prices_norm = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

            # Takens embedding: 1D series -> point cloud
            point_cloud = self.embedder.fit_transform(prices_norm.reshape(1, -1, 1))

            # Persistent homology
            diagrams = self.persistence.fit_transform(point_cloud)

            # Extract features from persistence diagrams
            entropy_vals = self.entropy.fit_transform(diagrams)[0]
            amplitude_val = self.amplitude.fit_transform(diagrams)[0][0]
            n_points_vals = self.n_points.fit_transform(diagrams)[0]

            # Count Betti numbers (connected components and loops)
            diagram = diagrams[0]
            h0_mask = diagram[:, 2] == 0
            h1_mask = diagram[:, 2] == 1

            h0_lifetimes = diagram[h0_mask, 1] - diagram[h0_mask, 0]
            h1_lifetimes = diagram[h1_mask, 1] - diagram[h1_mask, 0]

            # Filter out infinite lifetimes for counting
            h0_finite = h0_lifetimes[np.isfinite(h0_lifetimes)]
            h1_finite = h1_lifetimes[np.isfinite(h1_lifetimes)]

            betti_h0 = float(np.sum(h0_finite > LIFETIME_THRESHOLD))
            betti_h1 = float(np.sum(h1_finite > LIFETIME_THRESHOLD))

            n_sig_h0 = int(np.sum(h0_finite > LIFETIME_THRESHOLD * 2))
            n_sig_h1 = int(np.sum(h1_finite > LIFETIME_THRESHOLD * 2))

            complexity = betti_h0 + 2 * betti_h1 + amplitude_val

            return {
                "betti_h0": betti_h0,
                "betti_h1": betti_h1,
                "entropy_h0": float(entropy_vals[0]) if len(entropy_vals) > 0 else 0.0,
                "entropy_h1": float(entropy_vals[1]) if len(entropy_vals) > 1 else 0.0,
                "amplitude": float(amplitude_val),
                "n_significant_h0": n_sig_h0,
                "n_significant_h1": n_sig_h1,
                "complexity_score": float(complexity),
            }

        except Exception as e:
            logger.debug("TDA extraction failed: {}", e)
            return self._zero_features()

    def _zero_features(self) -> dict:
        """Return zero feature vector."""
        return {
            "betti_h0": 0.0, "betti_h1": 0.0,
            "entropy_h0": 0.0, "entropy_h1": 0.0,
            "amplitude": 0.0,
            "n_significant_h0": 0, "n_significant_h1": 0,
            "complexity_score": 0.0,
        }

    def extract_for_asset(self, asset: str, lookback_bars: int = 500) -> pd.DataFrame:
        """Extract TDA features for an asset using sliding windows."""
        con = duckdb.connect(self.db_path)
        df = con.execute("""
            SELECT timestamp, close FROM ohlcv
            WHERE asset = ? ORDER BY timestamp DESC LIMIT ?
        """, [asset, lookback_bars]).fetchdf()
        con.close()

        if len(df) < self.window_size:
            logger.warning("Not enough data for TDA on {}: {} bars", asset, len(df))
            return pd.DataFrame()

        df = df.sort_values("timestamp").reset_index(drop=True)
        prices = df["close"].values
        timestamps = df["timestamp"].values

        results = []
        for i in range(self.window_size, len(prices)):
            window = prices[i - self.window_size:i]
            features = self._extract_window_features(window)
            features["timestamp"] = timestamps[i]
            features["asset"] = asset
            results.append(features)

        return pd.DataFrame(results)

    def detect_phase_transitions(self, asset: str) -> list[dict]:
        """Detect sudden changes in Betti numbers signaling regime shifts."""
        con = duckdb.connect(self.db_path)
        df = con.execute("""
            SELECT * FROM tda_features WHERE asset = ?
            ORDER BY timestamp DESC LIMIT 200
        """, [asset]).fetchdf()
        con.close()

        if len(df) < 20:
            return []

        df = df.sort_values("timestamp").reset_index(drop=True)
        events = []

        for col in ["betti_h0", "betti_h1"]:
            series = df[col].values
            rolling_mean = pd.Series(series).rolling(10).mean().values
            rolling_std = pd.Series(series).rolling(10).std().values

            for i in range(11, len(series)):
                if rolling_std[i - 1] < 1e-10:
                    continue
                z = abs(series[i] - rolling_mean[i - 1]) / (rolling_std[i - 1] + 1e-10)
                if z > 3.0:
                    events.append({
                        "timestamp": df["timestamp"].iloc[i],
                        "asset": asset,
                        "event_type": f"phase_transition_{col}",
                        "betti_h0_before": float(rolling_mean[i - 1]) if col == "betti_h0" else 0.0,
                        "betti_h0_after": float(series[i]) if col == "betti_h0" else 0.0,
                        "betti_h1_before": float(rolling_mean[i - 1]) if col == "betti_h1" else 0.0,
                        "betti_h1_after": float(series[i]) if col == "betti_h1" else 0.0,
                        "description": f"{col} jumped from {rolling_mean[i-1]:.2f} to {series[i]:.2f} (z={z:.1f})",
                    })

        return events

    def run(self, assets: list[str] = None):
        """Run TDA pipeline on all assets and store results."""
        con = duckdb.connect(self.db_path)
        if assets is None:
            assets = [r[0] for r in con.execute("SELECT DISTINCT asset FROM ohlcv").fetchall()]
        con.close()

        all_features = []
        all_events = []

        for asset in assets:
            logger.info("Running TDA on {}", asset)
            features_df = self.extract_for_asset(asset)
            if not features_df.empty:
                all_features.append(features_df)

            events = self.detect_phase_transitions(asset)
            if events:
                all_events.extend(events)
                logger.info("Detected {} phase transitions for {}", len(events), asset)

        if all_features:
            combined = pd.concat(all_features, ignore_index=True)
            con = duckdb.connect(self.db_path)
            con.execute("BEGIN TRANSACTION")
            for _, row in combined.iterrows():
                con.execute("""
                    DELETE FROM tda_features WHERE timestamp = ? AND asset = ?
                """, [row["timestamp"], row["asset"]])
            con.execute("INSERT INTO tda_features SELECT * FROM combined")
            con.execute("COMMIT")
            con.close()
            logger.info("Stored {} TDA feature rows", len(combined))

        if all_events:
            events_df = pd.DataFrame(all_events)
            con = duckdb.connect(self.db_path)
            con.execute("INSERT INTO tda_events SELECT * FROM events_df")
            con.close()
            logger.info("Logged {} TDA events", len(events_df))

    def get_latest_features(self, asset: str, n: int = 1) -> pd.DataFrame:
        """Get the most recent TDA features for an asset."""
        con = duckdb.connect(self.db_path)
        df = con.execute("""
            SELECT * FROM tda_features WHERE asset = ?
            ORDER BY timestamp DESC LIMIT ?
        """, [asset, n]).fetchdf()
        con.close()
        return df
