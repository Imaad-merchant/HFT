"""Macro and fundamental layer — FRED data, economic calendar, news sentiment, regime classification."""

import numpy as np
import pandas as pd
import duckdb
import requests
import os
from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv

from macro.hmm_regime import HMMRegimeDetector, DEFAULT_PRIMARIES

load_dotenv()

DB_PATH = Path(__file__).parent.parent / "aria.db"

FRED_SERIES = {
    "FEDFUNDS": "fed_funds_rate",
    "CPIAUCSL": "cpi",
    "PCEPI": "pce",
    "GDP": "gdp",
    "UNRATE": "unemployment",
    "DGS10": "yield_10y",
    "DGS2": "yield_2y",
    "T10Y2Y": "yield_curve_spread",
}

HIGH_IMPACT_EVENTS = ["FOMC", "CPI", "NFP", "GDP", "PMI", "PPI", "Retail Sales", "PCE"]


class MacroEngine:
    """Pulls macro data, news sentiment, and classifies regimes."""

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)
        self.fred_key = os.getenv("FRED_API_KEY", "")
        self.news_key = os.getenv("NEWS_API_KEY", "")
        # One HMM per equity index primary (ES, NQ). Each is fit independently
        # with its own feature panel and labelled state map.
        self.hmm_models: dict[str, HMMRegimeDetector] = {
            asset: HMMRegimeDetector(self.db_path, primary_asset=asset)
            for asset in DEFAULT_PRIMARIES
        }
        self._init_db()

    def _init_db(self):
        """Create macro tables."""
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS macro_data (
                date DATE,
                series_id VARCHAR,
                series_name VARCHAR,
                value DOUBLE,
                PRIMARY KEY (date, series_id)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS economic_calendar (
                event_date TIMESTAMP,
                event_name VARCHAR,
                impact VARCHAR,
                actual VARCHAR,
                forecast VARCHAR,
                previous VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                timestamp TIMESTAMP,
                source VARCHAR,
                headline VARCHAR,
                sentiment_score DOUBLE,
                asset_tag VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS macro_regime (
                timestamp TIMESTAMP PRIMARY KEY,
                regime VARCHAR,
                es_signal DOUBLE,
                tlt_signal DOUBLE,
                vix_signal DOUBLE,
                dxy_signal DOUBLE,
                confidence DOUBLE
            )
        """)
        con.close()

    def fetch_fred_data(self):
        """Pull macro series from FRED API."""
        if not self.fred_key:
            logger.warning("No FRED_API_KEY — skipping FRED data")
            return

        con = duckdb.connect(self.db_path)
        for series_id, name in FRED_SERIES.items():
            try:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={series_id}&api_key={self.fred_key}"
                    f"&file_type=json&sort_order=desc&limit=500"
                )
                resp = requests.get(url, timeout=10)
                data = resp.json()

                if "observations" not in data:
                    logger.warning("No observations for {}", series_id)
                    continue

                rows = []
                for obs in data["observations"]:
                    if obs["value"] == ".":
                        continue
                    rows.append({
                        "date": obs["date"],
                        "series_id": series_id,
                        "series_name": name,
                        "value": float(obs["value"]),
                    })

                if rows:
                    df = pd.DataFrame(rows)
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    con.execute(f"DELETE FROM macro_data WHERE series_id = '{series_id}'")
                    con.execute("INSERT INTO macro_data SELECT * FROM df")
                    logger.info("Fetched {} observations for {} ({})", len(rows), series_id, name)

            except Exception as e:
                logger.error("FRED fetch failed for {}: {}", series_id, e)

        con.close()

    def fetch_news_sentiment(self):
        """Pull financial headlines and score sentiment."""
        if not self.news_key:
            logger.warning("No NEWS_API_KEY — skipping news sentiment")
            return

        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment not installed — skipping sentiment")
            return

        try:
            url = (
                f"https://newsapi.org/v2/everything"
                f"?q=stock+market+OR+federal+reserve+OR+treasury+OR+inflation"
                f"&language=en&sortBy=publishedAt&pageSize=50"
                f"&apiKey={self.news_key}"
            )
            resp = requests.get(url, timeout=10)
            data = resp.json()

            if data.get("status") != "ok":
                logger.warning("NewsAPI error: {}", data.get("message", "unknown"))
                return

            rows = []
            for article in data.get("articles", []):
                headline = article.get("title", "")
                if not headline:
                    continue

                scores = analyzer.polarity_scores(headline)
                sentiment = scores["compound"]

                # Tag by asset if possible
                asset_tag = "MACRO"
                headline_lower = headline.lower()
                if any(w in headline_lower for w in ["gold", "precious"]):
                    asset_tag = "GC"
                elif any(w in headline_lower for w in ["oil", "crude", "energy"]):
                    asset_tag = "CL"
                elif any(w in headline_lower for w in ["treasury", "bond", "yield"]):
                    asset_tag = "ZN"
                elif any(w in headline_lower for w in ["dollar", "usd", "currency"]):
                    asset_tag = "DXY"
                elif any(w in headline_lower for w in ["s&p", "nasdaq", "stock", "equity"]):
                    asset_tag = "ES"

                rows.append({
                    "timestamp": pd.Timestamp(article.get("publishedAt", datetime.now().isoformat())),
                    "source": article.get("source", {}).get("name", "unknown"),
                    "headline": headline[:500],
                    "sentiment_score": sentiment,
                    "asset_tag": asset_tag,
                })

            if rows:
                con = duckdb.connect(self.db_path)
                df = pd.DataFrame(rows)
                con.execute("INSERT INTO news_sentiment SELECT * FROM df")
                con.close()
                logger.info("Scored {} headlines", len(rows))

        except Exception as e:
            logger.error("News sentiment fetch failed: {}", e)

    def classify_regime(self) -> str:
        """Classify current macro regime from cross-asset signals."""
        con = duckdb.connect(self.db_path)

        # Get recent returns for regime classification
        regime_assets = {"ES": None, "TLT": None, "VIX": None, "DXY": None}
        for asset in regime_assets:
            df = con.execute("""
                SELECT close FROM ohlcv WHERE asset = ?
                ORDER BY timestamp DESC LIMIT 80
            """, [asset]).fetchdf()
            if len(df) >= 20:
                closes = df["close"].values[::-1]
                ret_20 = (closes[-1] / closes[-20] - 1) if closes[-20] != 0 else 0
                regime_assets[asset] = ret_20

        con.close()

        es = regime_assets.get("ES")
        tlt = regime_assets.get("TLT")
        vix = regime_assets.get("VIX")
        dxy = regime_assets.get("DXY")

        if None in (es, tlt, vix, dxy):
            return "NEUTRAL"

        # Regime classification
        if es > 0.01 and tlt < -0.005 and vix < -0.05 and dxy < 0:
            regime = "RISK_ON"
        elif es < -0.01 and tlt > 0.005 and vix > 0.05 and dxy > 0:
            regime = "RISK_OFF"
        elif es < 0 and regime_assets.get("GC") is not None:
            # Check for stagflation
            con = duckdb.connect(self.db_path)
            cl_df = con.execute(
                "SELECT close FROM ohlcv WHERE asset = 'CL' ORDER BY timestamp DESC LIMIT 80"
            ).fetchdf()
            gc_df = con.execute(
                "SELECT close FROM ohlcv WHERE asset = 'GC' ORDER BY timestamp DESC LIMIT 80"
            ).fetchdf()
            con.close()

            cl_up = False
            gc_up = False
            if len(cl_df) >= 20:
                cl_closes = cl_df["close"].values[::-1]
                cl_up = (cl_closes[-1] / cl_closes[-20] - 1) > 0.02
            if len(gc_df) >= 20:
                gc_closes = gc_df["close"].values[::-1]
                gc_up = (gc_closes[-1] / gc_closes[-20] - 1) > 0.02

            if cl_up and gc_up:
                regime = "STAGFLATION"
            else:
                regime = "NEUTRAL"
        else:
            regime = "NEUTRAL"

        # Store regime
        confidence = abs(es) + abs(tlt) + abs(vix) + abs(dxy)
        con = duckdb.connect(self.db_path)
        regime_row = pd.DataFrame([{
            "timestamp": pd.Timestamp.now(),
            "regime": regime,
            "es_signal": es, "tlt_signal": tlt,
            "vix_signal": vix, "dxy_signal": dxy,
            "confidence": confidence,
        }])
        con.execute("INSERT INTO macro_regime SELECT * FROM regime_row")
        con.close()

        logger.info("Macro regime: {} (confidence={:.4f})", regime, confidence)
        return regime

    def get_event_proximity(self) -> dict:
        """Get hours until next high-impact economic event."""
        con = duckdb.connect(self.db_path)
        now = pd.Timestamp.now()
        df = con.execute("""
            SELECT event_date, event_name FROM economic_calendar
            WHERE event_date > ? ORDER BY event_date LIMIT 5
        """, [now]).fetchdf()
        con.close()

        if df.empty:
            return {"hours_to_event": 999, "next_event": "NONE"}

        next_event = df.iloc[0]
        hours = (pd.Timestamp(next_event["event_date"]) - now).total_seconds() / 3600
        return {
            "hours_to_event": max(0, hours),
            "next_event": next_event["event_name"],
        }

    def get_sentiment_summary(self) -> dict:
        """Get aggregated sentiment by asset tag."""
        con = duckdb.connect(self.db_path)
        now = pd.Timestamp.now()
        since = now - timedelta(hours=24)
        df = con.execute("""
            SELECT asset_tag, AVG(sentiment_score) as avg_sentiment,
                   COUNT(*) as article_count
            FROM news_sentiment WHERE timestamp > ?
            GROUP BY asset_tag
        """, [since]).fetchdf()
        con.close()

        if df.empty:
            return {}

        return dict(zip(df["asset_tag"], df["avg_sentiment"]))

    def get_macro_features(self) -> dict:
        """Get latest macro features as a flat dict."""
        con = duckdb.connect(self.db_path)
        features = {}

        # Latest macro data values
        for series_id, name in FRED_SERIES.items():
            row = con.execute("""
                SELECT value FROM macro_data WHERE series_id = ?
                ORDER BY date DESC LIMIT 1
            """, [series_id]).fetchone()
            features[f"macro_{name}"] = float(row[0]) if row else 0.0

        # Yield curve slope
        y10 = features.get("macro_yield_10y", 0)
        y2 = features.get("macro_yield_2y", 0)
        features["macro_yield_curve_slope"] = y10 - y2

        # Rate of change of each macro variable
        for series_id, name in FRED_SERIES.items():
            rows = con.execute("""
                SELECT value FROM macro_data WHERE series_id = ?
                ORDER BY date DESC LIMIT 2
            """, [series_id]).fetchall()
            if len(rows) >= 2 and rows[1][0] != 0:
                features[f"macro_{name}_roc"] = (rows[0][0] - rows[1][0]) / abs(rows[1][0])
            else:
                features[f"macro_{name}_roc"] = 0.0

        # Current regime
        row = con.execute(
            "SELECT regime FROM macro_regime ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        features["macro_regime"] = row[0] if row else "NEUTRAL"

        # Event proximity
        event = self.get_event_proximity()
        features["macro_hours_to_event"] = event["hours_to_event"]

        # Sentiment
        sentiment = self.get_sentiment_summary()
        for tag in ["ES", "GC", "CL", "ZN", "DXY", "MACRO"]:
            features[f"sentiment_{tag}"] = sentiment.get(tag, 0.0)

        con.close()
        return features

    def run_hmm_forecast(self, refresh_visuals: bool = True) -> dict[str, dict]:
        """Run the HMM regime forecaster for every equity primary (ES, NQ).

        Lazily fits each model on first call. Safe to call repeatedly — the
        prediction step is cheap. Per-asset failures are isolated and logged
        so the rest of the macro pipeline keeps running. When `refresh_visuals`
        is True, also writes per-asset PNG/CSV trajectories to dashboard/.
        """
        out: dict[str, dict] = {}
        for asset, det in self.hmm_models.items():
            try:
                if not det.load():
                    logger.info("HMM[{}]: no saved model — fitting on available history", asset)
                    det.fit()
                fc = det.predict()
                logger.info(
                    "HMM[{}] | regime={} P(dump now)={:.1%} P(dump 1d)={:.1%} P(dump 2d)={:.1%}",
                    asset, fc.current_regime, fc.p_dump_now, fc.p_dump_1d, fc.p_dump_2d,
                )
                out[asset] = fc.to_dict()
                if refresh_visuals:
                    try:
                        det.plot_forecast(n_days=30)
                    except Exception as ve:
                        logger.debug("HMM[{}] visualization failed: {}", asset, ve)
            except Exception as e:
                logger.error("HMM[{}] forecast failed: {}", asset, e)
        return out

    def run(self):
        """Run full macro update cycle."""
        self.fetch_fred_data()
        self.fetch_news_sentiment()
        self.classify_regime()
        self.run_hmm_forecast()
        logger.info("Macro engine update complete")
