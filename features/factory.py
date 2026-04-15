"""Auto-generates 200+ candidate features from raw OHLCV data."""

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from pathlib import Path
from itertools import combinations

DB_PATH = Path(__file__).parent.parent / "aria.db"

RETURN_PERIODS = [1, 3, 5, 10, 20, 60]
VOLATILITY_PERIODS = [5, 10, 20, 60]
ZSCORE_PERIODS = [10, 20, 60]
CORRELATION_WINDOW = 60


class FeatureFactory:
    """Generates 200+ features per asset from OHLCV data."""

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)
        self._init_db()

    def _init_db(self):
        """Create features table."""
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS features (
                timestamp TIMESTAMP,
                asset VARCHAR,
                feature_name VARCHAR,
                value DOUBLE,
                PRIMARY KEY (timestamp, asset, feature_name)
            )
        """)
        # Wide-format features table for efficient RL consumption
        con.execute("""
            CREATE TABLE IF NOT EXISTS features_wide (
                timestamp TIMESTAMP,
                asset VARCHAR,
                feature_json VARCHAR,
                PRIMARY KEY (timestamp, asset)
            )
        """)
        con.close()

    def _compute_price_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Compute price-based features."""
        features = {}
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]

        # Returns over N periods
        for n in RETURN_PERIODS:
            features[f"ret_{n}"] = close.pct_change(n)

        # Rolling volatility
        for n in VOLATILITY_PERIODS:
            features[f"vol_{n}"] = close.pct_change().rolling(n).std()

        # Z-score of price relative to rolling mean
        for n in ZSCORE_PERIODS:
            roll_mean = close.rolling(n).mean()
            roll_std = close.rolling(n).std()
            features[f"zscore_{n}"] = (close - roll_mean) / (roll_std + 1e-10)

        # Price acceleration (second derivative)
        ret1 = close.pct_change()
        features["price_accel"] = ret1.diff()

        # Gap size
        features["gap"] = (open_ - close.shift(1)) / (close.shift(1) + 1e-10)

        # Intrabar range
        features["bar_range"] = (high - low) / (close + 1e-10)

        # Close position within bar
        features["close_position"] = (close - low) / (high - low + 1e-10)

        return features

    def _compute_volume_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Compute volume-based features."""
        features = {}
        volume = df["volume"].astype(float)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Volume z-score
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        features["vol_zscore"] = (volume - vol_mean) / (vol_std + 1e-10)

        # Volume delta
        features["vol_delta"] = volume.pct_change()

        # Price-volume divergence (price up + volume down = bearish divergence)
        price_dir = np.sign(close.pct_change())
        vol_dir = np.sign(volume.pct_change())
        features["pv_divergence"] = (price_dir - vol_dir) / 2.0

        # Trade intensity
        bar_range = high - low + 1e-10
        features["trade_intensity"] = volume / bar_range

        return features

    def _compute_regime_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Compute regime indicator features."""
        features = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Realized volatility regime
        rv = close.pct_change().rolling(20).std() * np.sqrt(252 * 78)  # Annualized from 5min bars
        rv_pct = rv.rolling(60).rank(pct=True)
        features["rv_regime"] = rv_pct

        # Trend strength (simplified ADX from OHLC)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        plus_dm = np.where((high - high.shift(1)) > (low.shift(1) - low), np.maximum(high - high.shift(1), 0), 0)
        minus_dm = np.where((low.shift(1) - low) > (high - high.shift(1)), np.maximum(low.shift(1) - low, 0), 0)
        atr = pd.Series(tr, index=df.index).rolling(14).mean()
        plus_di = pd.Series(plus_dm, index=df.index).rolling(14).mean() / (atr + 1e-10) * 100
        minus_di = pd.Series(minus_dm, index=df.index).rolling(14).mean() / (atr + 1e-10) * 100
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
        features["trend_strength"] = dx.rolling(14).mean()

        # Hurst exponent (simplified, rolling)
        features["hurst"] = self._rolling_hurst(close, window=60)

        return features

    def _rolling_hurst(self, series: pd.Series, window: int = 60) -> pd.Series:
        """Compute rolling Hurst exponent (R/S method)."""
        result = pd.Series(0.5, index=series.index)
        values = series.values

        for i in range(window, len(values)):
            window_data = values[i - window:i]
            returns = np.diff(window_data) / (window_data[:-1] + 1e-10)

            if len(returns) < 10:
                continue

            # Simplified R/S calculation
            mean_r = np.mean(returns)
            deviations = np.cumsum(returns - mean_r)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(returns) + 1e-10
            rs = R / S

            if rs > 0:
                result.iloc[i] = np.log(rs) / np.log(len(returns))

        return result

    def compute_cross_asset_features(self, all_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Compute cross-asset features (correlations, betas, spreads)."""
        # Align all assets by timestamp
        closes = {}
        for asset, df in all_data.items():
            s = df.set_index("timestamp")["close"]
            s.name = asset
            closes[asset] = s

        if not closes:
            return {}

        close_df = pd.DataFrame(closes)
        close_df = close_df.dropna(how="all").ffill()

        returns_df = close_df.pct_change()
        assets = list(closes.keys())
        cross_features = {a: {} for a in assets}

        # Rolling correlations for all pairs
        for a1, a2 in combinations(assets, 2):
            if a1 not in returns_df.columns or a2 not in returns_df.columns:
                continue
            corr = returns_df[a1].rolling(CORRELATION_WINDOW).corr(returns_df[a2])
            corr_mean = corr.rolling(200).mean()
            corr_std = corr.rolling(200).std()
            corr_z = (corr - corr_mean) / (corr_std + 1e-10)

            cross_features[a1][f"corr_{a2}"] = corr
            cross_features[a2][f"corr_{a1}"] = corr
            cross_features[a1][f"corr_z_{a2}"] = corr_z
            cross_features[a2][f"corr_z_{a1}"] = corr_z

        # Beta of each asset to ES
        if "ES" in returns_df.columns:
            es_var = returns_df["ES"].rolling(CORRELATION_WINDOW).var()
            for asset in assets:
                if asset == "ES":
                    continue
                cov = returns_df[asset].rolling(CORRELATION_WINDOW).cov(returns_df["ES"])
                cross_features[asset]["beta_to_es"] = cov / (es_var + 1e-10)

        # Spread z-scores for correlated pairs
        for a1, a2 in combinations(assets, 2):
            if a1 not in close_df.columns or a2 not in close_df.columns:
                continue
            spread = close_df[a1] / (close_df[a2] + 1e-10)
            spread_mean = spread.rolling(CORRELATION_WINDOW).mean()
            spread_std = spread.rolling(CORRELATION_WINDOW).std()
            spread_z = (spread - spread_mean) / (spread_std + 1e-10)
            cross_features[a1][f"spread_z_{a2}"] = spread_z

        return cross_features

    def compute_all_features(self, asset: str, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all single-asset features and return as wide DataFrame."""
        features = {}
        features.update(self._compute_price_features(df))
        features.update(self._compute_volume_features(df))
        features.update(self._compute_regime_features(df))

        result = pd.DataFrame(features, index=df.index)
        result["timestamp"] = df["timestamp"].values
        result["asset"] = asset
        return result

    def run(self, assets: list[str] = None):
        """Compute and store all features for all assets."""
        con = duckdb.connect(self.db_path)
        if assets is None:
            assets = [r[0] for r in con.execute("SELECT DISTINCT asset FROM ohlcv").fetchall()]

        all_data = {}
        for asset in assets:
            df = con.execute(
                "SELECT * FROM ohlcv WHERE asset = ? ORDER BY timestamp", [asset]
            ).fetchdf()
            if not df.empty:
                all_data[asset] = df
        con.close()

        # Compute single-asset features
        feature_frames = {}
        for asset, df in all_data.items():
            logger.info("Computing features for {}", asset)
            feature_frames[asset] = self.compute_all_features(asset, df)

        # Compute cross-asset features
        logger.info("Computing cross-asset features")
        cross_features = self.compute_cross_asset_features(all_data)

        # Merge cross-asset features into per-asset frames
        for asset in feature_frames:
            if asset in cross_features:
                for fname, series in cross_features[asset].items():
                    aligned = series.reindex(all_data[asset].set_index("timestamp").index)
                    feature_frames[asset][fname] = aligned.values

        # Store features as JSON blobs in wide format
        con = duckdb.connect(self.db_path)
        con.execute("BEGIN TRANSACTION")
        con.execute("DELETE FROM features_wide")

        for asset, fdf in feature_frames.items():
            cols = [c for c in fdf.columns if c not in ("timestamp", "asset")]
            for _, row in fdf.dropna(subset=["timestamp"]).iterrows():
                feat_dict = {c: float(row[c]) if pd.notna(row[c]) else 0.0 for c in cols}
                import json
                con.execute(
                    "INSERT OR REPLACE INTO features_wide VALUES (?, ?, ?)",
                    [row["timestamp"], asset, json.dumps(feat_dict)]
                )

        con.execute("COMMIT")
        con.close()
        total_features = sum(len([c for c in f.columns if c not in ("timestamp", "asset")]) for f in feature_frames.values())
        logger.info("Stored features for {} assets ({} total feature columns)", len(feature_frames), total_features)

    def get_feature_vector(self, asset: str, timestamp=None) -> dict:
        """Get the latest feature vector for an asset."""
        con = duckdb.connect(self.db_path)
        if timestamp:
            row = con.execute(
                "SELECT feature_json FROM features_wide WHERE asset = ? AND timestamp = ?",
                [asset, timestamp]
            ).fetchone()
        else:
            row = con.execute(
                "SELECT feature_json FROM features_wide WHERE asset = ? ORDER BY timestamp DESC LIMIT 1",
                [asset]
            ).fetchone()
        con.close()

        if row:
            import json
            return json.loads(row[0])
        return {}

    def get_feature_names(self) -> list[str]:
        """Get all feature names."""
        vec = self.get_feature_vector("ES")
        return list(vec.keys()) if vec else []
