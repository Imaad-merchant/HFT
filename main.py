"""ARIA — Adaptive Reinforcement Intelligence for Alpha.

Fully autonomous AI trading system. No user input required at runtime.

Usage:
    pip install -r requirements.txt
    python main.py
"""

import sys
import time
import threading
import schedule
from datetime import datetime
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"aria_{datetime.now().strftime('%Y%m%d')}.log"

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(str(log_file), level="DEBUG", rotation="100 MB", retention="7 days",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {module}:{function}:{line} | {message}")

# Import orchestrator after logging is configured
from agents.orchestrator import Orchestrator


def safe_run(func, name: str):
    """Run a function with crash protection — logs errors, never crashes the system."""
    def wrapper():
        try:
            func()
        except Exception as e:
            logger.error("{} failed: {} — will retry on next schedule", name, e)
    return wrapper


def main():
    """Initialize and run the full ARIA system."""
    logger.info("Starting ARIA...")

    # Initialize orchestrator
    orch = Orchestrator()

    # --- Phase 1: Initial data load ---
    logger.info("Phase 1: Loading market data...")
    try:
        orch.data_fetcher.fetch_all(period="2y")
        orch.data_fetcher.fill_gaps()
    except Exception as e:
        logger.warning("Initial data load partially failed: {} — continuing with available data", e)

    # --- Phase 2: Initial feature computation ---
    logger.info("Phase 2: Computing features...")
    try:
        orch.factory.run()
    except Exception as e:
        logger.warning("Feature computation failed: {}", e)

    try:
        orch.tda.run()
    except Exception as e:
        logger.warning("TDA pipeline failed: {}", e)

    # --- Phase 3: Initial macro update ---
    logger.info("Phase 3: Updating macro data...")
    try:
        orch.macro.run()
    except Exception as e:
        logger.warning("Macro update failed: {}", e)

    # --- Phase 4: Pair discovery ---
    logger.info("Phase 4: Discovering cointegrated pairs...")
    try:
        orch.stat_arb.discover_pairs()
    except Exception as e:
        logger.warning("Pair discovery failed: {}", e)

    # --- Phase 5: Print startup summary ---
    orch.startup_summary()

    # --- Schedule all recurring tasks ---

    # Every 5 minutes: data refresh + feature recompute
    schedule.every(5).minutes.do(safe_run(
        lambda: (orch.run_data_refresh(), orch.run_feature_recompute()),
        "DataRefresh+Features"
    ))

    # Every 5 minutes: run trading cycle
    schedule.every(5).minutes.do(safe_run(
        orch.run_trading_cycle,
        "TradingCycle"
    ))

    # Every 15 minutes: news sentiment update
    schedule.every(15).minutes.do(safe_run(
        orch.macro.fetch_news_sentiment,
        "NewsSentiment"
    ))

    # Every hour: macro regime + TDA + health update
    schedule.every(1).hours.do(safe_run(
        lambda: (orch.macro.classify_regime(), orch.run_tda(), orch.decay_monitor.run()),
        "HourlyUpdate"
    ))

    # Every day at 4:00 PM (market close): stat arb retest + decay report
    schedule.every().day.at("16:00").do(safe_run(
        lambda: (orch.stat_arb.retest_pairs(), orch.stat_arb.discover_pairs(), orch.run_decay_check()),
        "MarketCloseRoutine"
    ))

    # Every Sunday at 8:00 PM: evolutionary search
    schedule.every().sunday.at("20:00").do(safe_run(
        lambda: threading.Thread(target=orch.run_evolution, daemon=True).start(),
        "EvolutionSearch"
    ))

    # Every Sunday at 10:00 PM: full RL retraining
    schedule.every().sunday.at("22:00").do(safe_run(
        lambda: threading.Thread(target=lambda: orch.run_retrain(full=True), daemon=True).start(),
        "FullRetrain"
    ))

    # Every night at 11:00 PM: incremental fine-tuning
    schedule.every().day.at("23:00").do(safe_run(
        lambda: threading.Thread(target=lambda: orch.run_retrain(full=False), daemon=True).start(),
        "IncrementalFineTune"
    ))

    # --- Run the scheduler loop ---
    logger.info("ARIA is now running autonomously. All schedules active.")
    logger.info("Scheduled tasks:")
    for job in schedule.get_jobs():
        logger.info("  {}", job)

    while True:
        try:
            schedule.run_pending()
            time.sleep(10)
        except KeyboardInterrupt:
            logger.info("ARIA shutting down gracefully...")
            break
        except Exception as e:
            logger.error("Scheduler error: {} — restarting loop", e)
            time.sleep(30)


if __name__ == "__main__":
    main()
