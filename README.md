# ARIA — Adaptive Reinforcement Intelligence for Alpha

Fully autonomous AI trading system. Self-learning, self-adapting, discovers its own trading strategies from raw market data. Zero user input at runtime.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python main.py
```

## Required API Keys (.env)

- `FRED_API_KEY` — Get from https://fred.stlouisfed.org/docs/api/api_key.html
- `NEWS_API_KEY` — Get from https://newsapi.org/

Both are optional but recommended for full functionality.

## Architecture

```
aria/
  data/           Market data ingestion (yfinance -> DuckDB)
  features/       Feature engineering (200+ signals) + TDA pipeline
  environment/    Gymnasium trading simulator with realistic fills
  evolution/      Genetic strategy search (200 generations)
  rl/             PPO reinforcement learning agent (stable-baselines3)
  arbitrage/      Statistical arbitrage (cointegration, spread monitoring)
  macro/          FRED data, news sentiment, regime classification
  monitor/        Alpha decay detection and automated strategy retirement
  agents/         Orchestrator: signal fusion, risk gates, paper trading
  main.py         Headless entry point — runs everything autonomously
```

## What It Does

1. **Ingests** 2 years of 5-minute OHLCV data for 9 futures/ETFs
2. **Engineers** 200+ features per asset (price, volume, cross-asset, TDA)
3. **Discovers** cointegrated pairs for statistical arbitrage
4. **Evolves** rule-based strategies via genetic algorithms
5. **Trains** a PPO agent in a realistic trading simulator
6. **Classifies** macro regimes (Risk-On, Risk-Off, Stagflation)
7. **Fuses** all signals with Sharpe-weighted averaging
8. **Monitors** alpha decay and auto-retires dying strategies
9. **Paper trades** with full risk gates and circuit breakers

## Scheduling

| Interval | Task |
|---|---|
| 5 min | Data refresh + feature recompute + trading cycle |
| 15 min | News sentiment update |
| 1 hour | Macro regime + TDA + health snapshot |
| Daily 4pm | Stat arb retest + pair discovery + decay report |
| Sunday 8pm | Evolutionary search (200 generations) |
| Sunday 10pm | Full RL retraining (2M steps) |
| Nightly 11pm | Incremental RL fine-tuning (10k steps) |

## Risk Controls

- Max position per asset: 20% of capital
- Max portfolio exposure: 100% of capital
- Circuit breaker: halts all trading if drawdown > 8%
- RISK_OFF regime: reduces all positions by 50%
- Event proximity: goes flat 30 minutes before high-impact events
- Alpha decay: auto-reduces allocation when 7-day Sharpe < 0.3, suspends at < 0.0

## Data Storage

All data stored in `aria.db` (DuckDB). Key tables: `ohlcv`, `features_wide`, `tda_features`, `pairs`, `trades`, `signals`, `strategy_health`, `alpha_events`, `macro_regime`, `retraining_log`.

Health snapshot written hourly to `aria_health.json`.

## Logs

Written to `logs/aria_YYYYMMDD.log` with automatic rotation.
