"""Build a static HMM dashboard site for GitHub Pages.

Designed to run in a GitHub Actions workflow (or any environment with
internet access). Fetches market data, fits the per-asset HMMs, renders
forecast PNGs, and writes a self-contained mobile-friendly index.html plus
a JSON snapshot to dashboard/site/. The site is then uploaded as a Pages
artifact and deployed by the workflow.

CLI:
    python -m dashboard.build_static                       # default behaviour
    python -m dashboard.build_static --no-fetch            # use existing aria.db
    python -m dashboard.build_static --asset ES --days 60
    python -m dashboard.build_static --out path/to/dir
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# Make sibling project modules importable when invoked as `python -m dashboard.build_static`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.fetcher import DataFetcher  # noqa: E402
from macro.macro import MacroEngine  # noqa: E402
from macro.hmm_regime import HMMRegimeDetector, DEFAULT_PRIMARIES  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).parent / "site"
DEFAULT_DAYS = 30


# --------------------------------------------------------------------- helpers


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _fmt_signed_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def _fmt_first_passage(d: float | None) -> str:
    if d is None:
        return "unreachable"
    if d == 0.0:
        return "in dump now"
    return f"~{d:.1f}d"


def _render_index_html(payload: dict, days: int) -> str:
    forecasts: dict[str, dict] = payload["forecasts"]

    rows: list[str] = []
    for asset, fc in forecasts.items():
        if "error" in fc:
            rows.append(
                f'<tr><td>{asset}</td><td colspan="6">error: {fc["error"]}</td></tr>'
            )
            continue
        regime_class = fc["current_regime"].lower()
        rows.append(
            "<tr>"
            f'<td><strong>{asset}</strong></td>'
            f'<td><span class="regime regime-{regime_class}">{fc["current_regime"]}</span></td>'
            f'<td>{_fmt_pct(fc["p_dump_now"])}</td>'
            f'<td>{_fmt_pct(fc["p_dump_1d"])}</td>'
            f'<td>{_fmt_pct(fc["p_dump_2d"])}</td>'
            f'<td>{_fmt_signed_pct(fc["expected_1d_return"])}</td>'
            f'<td>{_fmt_first_passage(fc.get("expected_first_dump_days"))}</td>'
            "</tr>"
        )

    chart_blocks: list[str] = []
    for asset in forecasts:
        if "error" in forecasts[asset]:
            continue
        chart_blocks.append(
            '<section class="chart">'
            f'  <h2>{asset}</h2>'
            f'  <img src="hmm_forecast_{asset.lower()}.png" alt="{asset} HMM forecast">'
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta http-equiv="refresh" content="300">
  <title>ARIA HMM Regime Forecast</title>
  <style>
    :root {{ color-scheme: dark; }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif;
      background: #0f1216; color: #e7eaee;
      max-width: 1100px; margin: 0 auto; padding: 16px;
      -webkit-font-smoothing: antialiased;
    }}
    h1 {{ font-weight: 600; margin: 0 0 4px 0; font-size: 22px; }}
    header p {{ color: #8e98a6; margin: 0 0 20px 0; font-size: 13px; }}
    .table-wrap {{ overflow-x: auto; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 560px;
             background: #161b22; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; text-align: left;
              border-bottom: 1px solid #222932; font-size: 14px; }}
    th {{ background: #1c232c; font-weight: 500; color: #b6c0cb;
          font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    td {{ font-variant-numeric: tabular-nums; }}
    tr:last-child td {{ border-bottom: none; }}
    .regime {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
               font-size: 11px; font-weight: 700; letter-spacing: 0.04em; }}
    .regime-bull   {{ background: #1e3a26; color: #6dd47e; }}
    .regime-normal {{ background: #3a3416; color: #f5d76e; }}
    .regime-bear   {{ background: #3d2718; color: #f0a060; }}
    .regime-crash  {{ background: #3a181a; color: #ff6b6b; }}
    section.chart {{ margin: 16px 0; padding: 14px; background: #161b22;
                     border-radius: 8px; }}
    section.chart h2 {{ margin: 0 0 10px 0; font-weight: 600; color: #e7eaee;
                        font-size: 16px; }}
    section.chart img {{ width: 100%; height: auto; display: block;
                         border-radius: 6px; }}
    footer {{ color: #8e98a6; font-size: 12px; margin-top: 28px;
              line-height: 1.55; padding-bottom: 32px; }}
    a {{ color: #8ec5ff; }}
    code {{ background: #1c232c; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>ARIA HMM Regime Forecast</h1>
    <p>Generated {payload['generated_at']} &middot; horizon {days} trading days</p>
  </header>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Asset</th><th>Regime</th>
          <th>P(now)</th><th>P(1d)</th><th>P(2d)</th>
          <th>E[ret&nbsp;1d]</th><th>E[first&nbsp;dump]</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
  {''.join(chart_blocks)}
  <footer>
    HMM dump probabilities are conditional similarity scores from a fitted
    Gaussian HMM, not directional crash calls. They tell you how much the
    current tape resembles historical stress regimes and how the empirical
    transition matrix rolls forward. Treat them as a risk dial alongside the
    macro regime and event-proximity gates already in the orchestrator.
    <br><br>
    Raw data: <a href="forecast.json">forecast.json</a>
  </footer>
</body>
</html>
"""


# ----------------------------------------------------------------------- build


def build(
    out_dir: Path = DEFAULT_OUT_DIR,
    assets: list[str] | None = None,
    days: int = DEFAULT_DAYS,
    fetch_data: bool = True,
) -> Path:
    """Build the static HMM dashboard site. Returns the output directory path."""
    assets = list(assets or DEFAULT_PRIMARIES)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Refresh OHLCV from yfinance
    if fetch_data:
        logger.info("Fetching market data via yfinance...")
        try:
            fetcher = DataFetcher()
            fetcher.fetch_all(period="2y")
            fetcher.fill_gaps()
        except Exception as e:
            logger.warning("Market data fetch failed: {} — continuing with whatever's in aria.db", e)

    # 2. Refresh FRED + news sentiment if API keys are set as env vars / repo secrets
    me = MacroEngine()
    try:
        me.fetch_fred_data()
    except Exception as e:
        logger.warning("FRED fetch failed: {}", e)
    try:
        me.fetch_news_sentiment()
    except Exception as e:
        logger.warning("News sentiment fetch failed: {}", e)

    # 3. Fit + predict + render per asset
    forecasts: dict[str, dict] = {}
    for asset in assets:
        try:
            det = HMMRegimeDetector(primary_asset=asset)
            # Always refit on a fresh CI run — data has changed since the last build
            det.fit()
            fc = det.predict()
            payload = fc.to_dict()
            payload["expected_first_dump_days"] = det.expected_first_passage_to_dump()
            forecasts[asset] = payload
            det.plot_forecast(n_days=days, output_dir=out_dir)
            logger.info(
                "HMM[{}] | regime={} P(now)={:.1%} P(1d)={:.1%} P(2d)={:.1%}",
                asset, fc.current_regime, fc.p_dump_now, fc.p_dump_1d, fc.p_dump_2d,
            )
        except Exception as e:
            logger.error("HMM[{}] failed: {}", asset, e)
            forecasts[asset] = {"error": str(e)}

    # 4. Write index.html and forecast.json
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_days": days,
        "forecasts": forecasts,
    }
    (out_dir / "index.html").write_text(_render_index_html(payload, days))
    (out_dir / "forecast.json").write_text(json.dumps(payload, indent=2, default=str))

    logger.info("Static site built at {}", out_dir)
    logger.info("Files: {}", sorted(p.name for p in out_dir.iterdir()))
    return out_dir


# ------------------------------------------------------------------------- CLI


def _parse_args(argv: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "out_dir": DEFAULT_OUT_DIR,
        "assets": list(DEFAULT_PRIMARIES),
        "days": DEFAULT_DAYS,
        "fetch_data": True,
    }
    i = 1
    while i < len(argv):
        tok = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if tok == "--out" and nxt is not None:
            out["out_dir"] = Path(nxt); i += 2; continue
        if tok == "--asset" and nxt is not None:
            out["assets"] = [a.strip().upper() for a in nxt.split(",") if a.strip()]
            i += 2; continue
        if tok == "--days" and nxt is not None:
            out["days"] = int(nxt); i += 2; continue
        if tok == "--no-fetch":
            out["fetch_data"] = False; i += 1; continue
        if tok in ("-h", "--help"):
            print(__doc__); sys.exit(0)
        i += 1
    return out


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    build(**args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
