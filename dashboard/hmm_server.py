"""Tiny localhost web UI for the HMM regime forecaster.

A self-contained HTTP server (stdlib only — no Flask/FastAPI) that exposes the
ARIA HMM regime forecasts as a single auto-refreshing dark-mode page plus a
small JSON API. Each request regenerates the forecast from the current state
of `aria.db`, so the page always reflects the latest data the system has
ingested.

Routes:
  GET /                           -> HTML dashboard (auto-refresh 60s)
  GET /api/forecast               -> JSON: all asset forecasts
  GET /api/forecast/<asset>       -> JSON: single asset forecast
  GET /chart/<asset>.png          -> PNG: per-asset 3-panel chart
  GET /healthz                    -> plain text "ok"

CLI:
    python -m dashboard.hmm_server
    python -m dashboard.hmm_server --port 9000
    python -m dashboard.hmm_server --asset ES --days 60
    python -m dashboard.hmm_server --host 0.0.0.0  # expose on LAN

The server is single-threaded by default, which is fine for a personal
dashboard refreshing once a minute. Visualisation regeneration takes ~1s per
asset; if you point a benchmark at it you can add caching.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

from loguru import logger

from macro.hmm_regime import HMMRegimeDetector, DEFAULT_PRIMARIES

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_DAYS = 30
CHART_DIR = Path(__file__).parent

# Mutable server config populated at startup. Single-process, single-threaded
# server, so a plain module-level dict is fine.
_CONFIG: dict[str, Any] = {
    "assets": list(DEFAULT_PRIMARIES),
    "days": DEFAULT_DAYS,
    "db_path": None,
}


# --------------------------------------------------------------------- helpers


def _get_or_fit(asset: str) -> HMMRegimeDetector:
    """Load the saved per-asset HMM, fitting it on demand the first time."""
    det = HMMRegimeDetector(db_path=_CONFIG["db_path"], primary_asset=asset)
    if not det.load():
        logger.info("HMM[{}]: no saved model — fitting now", asset)
        det.fit()
    return det


def _generate_chart_bytes(asset: str, days: int) -> bytes:
    """Render the per-asset forecast PNG and return its raw bytes."""
    det = _get_or_fit(asset)
    det.plot_forecast(n_days=days, output_dir=CHART_DIR)
    png_path = CHART_DIR / f"hmm_forecast_{asset.lower()}.png"
    return png_path.read_bytes()


def _build_forecast_payload() -> dict:
    """Build the cross-asset forecast payload used by both the HTML and JSON routes."""
    forecasts: dict[str, dict] = {}
    for a in _CONFIG["assets"]:
        try:
            det = _get_or_fit(a)
            fc = det.predict()
            payload = fc.to_dict()
            payload["expected_first_dump_days"] = det.expected_first_passage_to_dump()
            forecasts[a] = payload
        except Exception as e:
            logger.error("HMM[{}] forecast failed: {}", a, e)
            forecasts[a] = {"error": str(e)}
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "horizon_days": _CONFIG["days"],
        "forecasts": forecasts,
    }


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


def _render_index_html(payload: dict) -> str:
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

    cache_buster = int(datetime.utcnow().timestamp())
    chart_blocks: list[str] = []
    for asset in forecasts:
        chart_blocks.append(
            '<section class="chart">'
            f'  <h2>{asset}</h2>'
            f'  <img src="/chart/{asset.lower()}.png?days={_CONFIG["days"]}&_={cache_buster}" '
            f'       alt="{asset} HMM forecast">'
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="refresh" content="60">
  <title>ARIA HMM Regime Forecast</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      background: #0f1216; color: #e7eaee;
      max-width: 1200px; margin: 24px auto; padding: 0 16px;
    }}
    h1 {{ font-weight: 600; margin: 0 0 4px 0; font-size: 22px; }}
    header p {{ color: #8e98a6; margin: 0 0 24px 0; font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px;
             background: #161b22; border-radius: 6px; overflow: hidden; }}
    th, td {{ padding: 10px 14px; text-align: left;
              border-bottom: 1px solid #222932; font-size: 14px; }}
    th {{ background: #1c232c; font-weight: 500; color: #b6c0cb; }}
    td {{ font-variant-numeric: tabular-nums; }}
    tr:last-child td {{ border-bottom: none; }}
    .regime {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
               font-size: 12px; font-weight: 600; }}
    .regime-bull   {{ background: #1e3a26; color: #6dd47e; }}
    .regime-normal {{ background: #3a3416; color: #f5d76e; }}
    .regime-bear   {{ background: #3d2718; color: #f0a060; }}
    .regime-crash  {{ background: #3a181a; color: #ff6b6b; }}
    section.chart {{ margin: 24px 0; padding: 16px; background: #161b22;
                     border-radius: 6px; }}
    section.chart h2 {{ margin: 0 0 12px 0; font-weight: 500; color: #b6c0cb;
                        font-size: 16px; }}
    section.chart img {{ width: 100%; height: auto; display: block;
                         border-radius: 4px; }}
    footer {{ color: #8e98a6; font-size: 12px; margin-top: 32px;
              line-height: 1.5; padding-bottom: 24px; }}
    code {{ background: #1c232c; padding: 1px 6px; border-radius: 3px;
            font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>ARIA HMM Regime Forecast</h1>
    <p>Generated {payload['generated_at']} &middot;
       horizon {payload['horizon_days']} trading days &middot;
       page auto-refreshes every 60s</p>
  </header>
  <table>
    <thead>
      <tr>
        <th>Asset</th><th>Regime</th>
        <th>P(dump now)</th><th>P(dump 1d)</th><th>P(dump 2d)</th>
        <th>E[ret 1d]</th><th>E[first dump]</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  {''.join(chart_blocks)}
  <footer>
    HMM dump probabilities are conditional similarity scores from a fitted
    Gaussian HMM, not directional crash calls. They tell you how much the
    current tape resembles historical stress regimes and how the empirical
    transition matrix rolls forward. Treat them as a risk dial alongside the
    macro regime and event-proximity gates already in the orchestrator.
    <br><br>
    API:
      <code>GET /api/forecast</code>,
      <code>GET /api/forecast/&lt;asset&gt;</code>,
      <code>GET /chart/&lt;asset&gt;.png?days=N</code>
  </footer>
</body>
</html>
"""


# ---------------------------------------------------------------- HTTP handler


class HMMHandler(BaseHTTPRequestHandler):
    server_version = "ARIA-HMM/1.0"

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        self._send_bytes(status, html.encode("utf-8"), "text/html; charset=utf-8")

    def _send_json(self, status: int, obj: Any) -> None:
        body = json.dumps(obj, indent=2, default=str).encode("utf-8")
        self._send_bytes(status, body, "application/json")

    def _send_png(self, body: bytes) -> None:
        self._send_bytes(200, body, "image/png")

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler convention)
        try:
            url = urlparse(self.path)
            qs = parse_qs(url.query)
            path = url.path

            if path == "/" or path == "":
                self._send_html(200, _render_index_html(_build_forecast_payload()))
                return

            if path == "/healthz":
                self._send_bytes(200, b"ok", "text/plain")
                return

            if path == "/api/forecast":
                self._send_json(200, _build_forecast_payload())
                return

            if path.startswith("/api/forecast/"):
                asset = path.rsplit("/", 1)[1].upper()
                if asset not in _CONFIG["assets"]:
                    self._send_json(404, {"error": f"unknown asset {asset}",
                                          "known": _CONFIG["assets"]})
                    return
                det = _get_or_fit(asset)
                fc = det.predict()
                payload = fc.to_dict()
                payload["expected_first_dump_days"] = det.expected_first_passage_to_dump()
                self._send_json(200, payload)
                return

            if path.startswith("/chart/") and path.endswith(".png"):
                asset = path[len("/chart/") : -len(".png")].upper()
                if asset not in _CONFIG["assets"]:
                    self._send_html(404, f"<h1>404</h1><p>Unknown asset {asset}</p>")
                    return
                try:
                    days = int(qs.get("days", [_CONFIG["days"]])[0])
                except ValueError:
                    days = _CONFIG["days"]
                self._send_png(_generate_chart_bytes(asset, days))
                return

            self._send_html(404, "<h1>404</h1><p>Not found.</p>")

        except Exception as e:
            logger.exception("HMM server: request failed: {}", e)
            try:
                self._send_html(500, f"<h1>500</h1><pre>{type(e).__name__}: {e}</pre>")
            except Exception:
                pass

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("HTTP {} - {}", self.address_string(), format % args)


# ---------------------------------------------------------------------- serve


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    assets: list[str] | None = None,
    days: int = DEFAULT_DAYS,
    db_path: str | None = None,
) -> None:
    _CONFIG["assets"] = list(assets or DEFAULT_PRIMARIES)
    _CONFIG["days"] = days
    _CONFIG["db_path"] = db_path

    httpd = HTTPServer((host, port), HMMHandler)
    url = f"http://{host}:{port}/"
    logger.info(
        "HMM dashboard ready: {} (assets={}, horizon={}d)",
        url, _CONFIG["assets"], days,
    )
    print(f"\n  HMM dashboard:  {url}\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("HMM dashboard: shutting down")
    finally:
        httpd.server_close()


# --------------------------------------------------------------- arg parsing


def _parse_args(argv: list[str]) -> dict:
    out: dict[str, Any] = {
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "assets": list(DEFAULT_PRIMARIES),
        "days": DEFAULT_DAYS,
        "db_path": None,
    }
    i = 1
    while i < len(argv):
        tok = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if tok == "--port" and nxt is not None:
            out["port"] = int(nxt); i += 2; continue
        if tok == "--host" and nxt is not None:
            out["host"] = nxt; i += 2; continue
        if tok == "--asset" and nxt is not None:
            out["assets"] = [a.strip().upper() for a in nxt.split(",") if a.strip()]
            i += 2; continue
        if tok == "--days" and nxt is not None:
            out["days"] = int(nxt); i += 2; continue
        if tok == "--db" and nxt is not None:
            out["db_path"] = nxt; i += 2; continue
        if tok in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        i += 1
    return out


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    serve(**args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
