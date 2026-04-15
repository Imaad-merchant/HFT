"""Topological Data Analysis — persistent homology for regime detection and grid topology."""

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude, NumberOfPoints
    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False
    logger.warning("giotto-tda not installed — TDA features will use scipy fallback")

DB_PATH = Path(__file__).parent.parent / "aria.db"

WINDOW_SIZE = 60
TAKENS_DIM = 3
TAKENS_DELAY = 5
LIFETIME_THRESHOLD = 0.05


def takens_embedding(series, dim=3, delay=5):
    """Manual Takens time-delay embedding: 1D series -> point cloud in R^dim."""
    n = len(series) - (dim - 1) * delay
    if n <= 0:
        return np.array([]).reshape(0, dim)
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = series[i * delay: i * delay + n]
    return embedded


def compute_persistence_scipy(point_cloud):
    """Compute persistence diagram using scipy distance matrix (fallback when gtda fails)."""
    if len(point_cloud) < 3:
        return {"h0_lifetimes": np.array([]), "h1_lifetimes": np.array([]), "birth_death_h0": [], "birth_death_h1": []}

    # Compute pairwise distances
    dists = squareform(pdist(point_cloud))

    # Simplified Rips filtration for H0 (connected components via single-linkage)
    n = len(point_cloud)
    # Sort all edges by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dists[i, j], i, j))
    edges.sort()

    # Union-Find for H0
    parent = list(range(n))
    birth = [0.0] * n  # All components born at 0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    h0_lifetimes = []
    h0_birth_death = []
    for dist, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            # Merge: the younger component dies
            h0_lifetimes.append(dist - 0.0)
            h0_birth_death.append((0.0, dist))
            parent[ri] = rj

    # Simplified H1: detect loops via short cycles in distance matrix
    h1_lifetimes = []
    h1_birth_death = []
    threshold = np.percentile(dists[dists > 0], 50) if np.any(dists > 0) else 1.0
    cap = min(n, 20)  # Cap at 20 points for O(n³) performance
    for i in range(cap):
        for j in range(i + 1, cap):
            for k in range(j + 1, cap):
                sides = sorted([dists[i, j], dists[j, k], dists[i, k]])
                birth_val = sides[1]  # Triangle appears when 2nd longest edge added
                death_val = sides[2]  # Triangle filled when longest edge added
                if death_val - birth_val > LIFETIME_THRESHOLD * threshold:
                    h1_lifetimes.append(death_val - birth_val)
                    h1_birth_death.append((birth_val, death_val))

    return {
        "h0_lifetimes": np.array(h0_lifetimes),
        "h1_lifetimes": np.array(h1_lifetimes[:50]),  # Cap for performance
        "birth_death_h0": h0_birth_death,
        "birth_death_h1": h1_birth_death[:50],
    }


class TDAFeatureExtractor:
    """Extracts topological features from price series via persistent homology."""

    def __init__(self, db_path=None, window_size=WINDOW_SIZE):
        self.db_path = str(db_path or DB_PATH)
        self.window_size = window_size
        self._init_db()

        if GTDA_AVAILABLE:
            self.persistence = VietorisRipsPersistence(
                homology_dimensions=[0, 1],
                max_edge_length=np.inf,
                n_jobs=1
            )
            self.entropy = PersistenceEntropy()
            self.amplitude = Amplitude(metric="landscape")

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

    def _extract_window_features(self, prices):
        """Extract TDA features from a single price window."""
        if len(prices) < self.window_size:
            return self._zero_features()

        try:
            # Normalize prices
            prices_norm = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

            # Takens embedding: 1D -> point cloud in R^3
            point_cloud = takens_embedding(prices_norm, dim=TAKENS_DIM, delay=TAKENS_DELAY)
            if len(point_cloud) < 5:
                return self._zero_features()

            # Try giotto-tda first, fall back to scipy
            if GTDA_AVAILABLE:
                try:
                    pc_3d = point_cloud.reshape(1, *point_cloud.shape)
                    diagrams = self.persistence.fit_transform(pc_3d)
                    entropy_vals = self.entropy.fit_transform(diagrams)[0]
                    amplitude_val = float(self.amplitude.fit_transform(diagrams)[0][0])

                    diagram = diagrams[0]
                    h0_mask = diagram[:, 2] == 0
                    h1_mask = diagram[:, 2] == 1
                    h0_lt = diagram[h0_mask, 1] - diagram[h0_mask, 0]
                    h1_lt = diagram[h1_mask, 1] - diagram[h1_mask, 0]
                    h0_finite = h0_lt[np.isfinite(h0_lt)]
                    h1_finite = h1_lt[np.isfinite(h1_lt)]

                    return self._build_features(h0_finite, h1_finite,
                                                 float(entropy_vals[0]) if len(entropy_vals) > 0 else 0.0,
                                                 float(entropy_vals[1]) if len(entropy_vals) > 1 else 0.0,
                                                 amplitude_val)
                except Exception:
                    pass  # Fall through to scipy

            # Scipy fallback
            result = compute_persistence_scipy(point_cloud)
            h0_lt = result["h0_lifetimes"]
            h1_lt = result["h1_lifetimes"]

            # Compute entropy manually
            def persistence_entropy(lifetimes):
                if len(lifetimes) == 0:
                    return 0.0
                L = lifetimes / (lifetimes.sum() + 1e-10)
                L = L[L > 0]
                return float(-np.sum(L * np.log(L + 1e-10)))

            return self._build_features(
                h0_lt, h1_lt,
                persistence_entropy(h0_lt),
                persistence_entropy(h1_lt),
                float(np.sum(h0_lt) + np.sum(h1_lt))
            )

        except Exception as e:
            logger.debug("TDA extraction failed: {}", e)
            return self._zero_features()

    def _build_features(self, h0_lt, h1_lt, entropy_h0, entropy_h1, amplitude):
        """Build feature dict from persistence data."""
        betti_h0 = float(np.sum(h0_lt > LIFETIME_THRESHOLD)) if len(h0_lt) > 0 else 0.0
        betti_h1 = float(np.sum(h1_lt > LIFETIME_THRESHOLD)) if len(h1_lt) > 0 else 0.0
        n_sig_h0 = int(np.sum(h0_lt > LIFETIME_THRESHOLD * 2)) if len(h0_lt) > 0 else 0
        n_sig_h1 = int(np.sum(h1_lt > LIFETIME_THRESHOLD * 2)) if len(h1_lt) > 0 else 0
        complexity = betti_h0 + 2 * betti_h1 + amplitude

        return {
            "betti_h0": betti_h0,
            "betti_h1": betti_h1,
            "entropy_h0": entropy_h0,
            "entropy_h1": entropy_h1,
            "amplitude": amplitude,
            "n_significant_h0": n_sig_h0,
            "n_significant_h1": n_sig_h1,
            "complexity_score": float(complexity),
        }

    def _zero_features(self):
        """Return zero feature vector."""
        return {
            "betti_h0": 0.0, "betti_h1": 0.0,
            "entropy_h0": 0.0, "entropy_h1": 0.0,
            "amplitude": 0.0,
            "n_significant_h0": 0, "n_significant_h1": 0,
            "complexity_score": 0.0,
        }

    def extract_topological_levels(self, prices):
        """Extract support/resistance levels from H0 persistence diagram.

        In the Takens point cloud, long-lived H0 components correspond to
        price clusters — natural support and resistance zones. The birth/death
        values map back to price levels where the market spends significant time.
        """
        if len(prices) < self.window_size:
            return [], 0.0

        prices_norm = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)
        point_cloud = takens_embedding(prices_norm, dim=TAKENS_DIM, delay=TAKENS_DELAY)

        if len(point_cloud) < 5:
            return [], 0.0

        result = compute_persistence_scipy(point_cloud)

        # Map H0 birth-death pairs back to price levels
        # Each connected component's merge point indicates a price cluster boundary
        price_range = prices.max() - prices.min()
        price_min = prices.min()
        levels = []

        for birth, death in result["birth_death_h0"]:
            lifetime = death - birth
            if lifetime > LIFETIME_THRESHOLD:
                # Map the merge distance back to a price level
                # Use the midpoint of the merge as the level
                level_norm = (birth + death) / 2.0
                # Scale back to price space using the point cloud centroid
                level_price = price_min + level_norm * price_range
                levels.append({
                    "price": float(level_price),
                    "strength": float(lifetime),
                    "type": "support" if level_price < prices[-1] else "resistance",
                })

        # Sort by strength (persistence)
        levels.sort(key=lambda x: x["strength"], reverse=True)

        # H1 cyclicality score: more loops = more cyclical = better for grids
        h1_lt = result["h1_lifetimes"]
        cyclicality = float(np.sum(h1_lt > LIFETIME_THRESHOLD)) if len(h1_lt) > 0 else 0.0

        return levels[:10], cyclicality  # Top 10 levels

    def is_grid_favorable(self, prices):
        """Determine if market topology favors grid trading.

        Grid trading works when:
        - H1 (loops) is high: price is cycling, mean-reverting
        - H0 clusters are well-defined: clear support/resistance
        - Complexity is moderate: not trending, not chaotic
        """
        features = self._extract_window_features(prices)

        betti_h1 = features["betti_h1"]
        complexity = features["complexity_score"]
        entropy_h1 = features["entropy_h1"]

        # High H1 = cyclical patterns = good for grids
        # Moderate complexity = range-bound = good for grids
        # Low entropy = predictable cycles = good for grids
        score = 0.0
        if betti_h1 >= 2:
            score += 0.4  # Cycles detected
        if 2 < complexity < 15:
            score += 0.3  # Moderate complexity
        if entropy_h1 < 1.5:
            score += 0.3  # Predictable

        return score, features

    def extract_for_asset(self, asset, lookback_bars=500):
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
        # Process every 5th window for performance (still captures regime changes)
        for i in range(self.window_size, len(prices), 5):
            window = prices[i - self.window_size:i]
            features = self._extract_window_features(window)
            features["timestamp"] = timestamps[i]
            features["asset"] = asset
            results.append(features)

        return pd.DataFrame(results) if results else pd.DataFrame()

    def detect_phase_transitions(self, asset):
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

    def run(self, assets=None):
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
                logger.info("  {} TDA windows for {}", len(features_df), asset)

            events = self.detect_phase_transitions(asset)
            if events:
                all_events.extend(events)
                logger.info("  {} phase transitions for {}", len(events), asset)

        if all_features:
            combined = pd.concat(all_features, ignore_index=True)
            col_order = ["timestamp", "asset", "betti_h0", "betti_h1",
                         "entropy_h0", "entropy_h1", "amplitude",
                         "n_significant_h0", "n_significant_h1", "complexity_score"]
            combined = combined[col_order]
            con = duckdb.connect(self.db_path)
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM tda_features")
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

    def get_latest_features(self, asset, n=1):
        """Get the most recent TDA features for an asset."""
        con = duckdb.connect(self.db_path)
        df = con.execute("""
            SELECT * FROM tda_features WHERE asset = ?
            ORDER BY timestamp DESC LIMIT ?
        """, [asset, n]).fetchdf()
        con.close()
        return df
