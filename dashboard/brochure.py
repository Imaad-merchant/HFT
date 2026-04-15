"""Brochure-style composite forecast image for the HMM regime dashboard.

Produces a single portrait-orientation PNG that combines the current regime
status of every primary asset (ES, NQ) into one shareable, phone-friendly
image. Layout:

    +---------------------------------+
    |  TITLE BAR                      |
    +---------------------------------+
    |  ┌─────────┐ ┌─────────┐        |
    |  │  ES     │ │  NQ     │ Status |
    |  │ REGIME  │ │ REGIME  │ cards  |
    |  │ stats   │ │ stats   │        |
    |  └─────────┘ └─────────┘        |
    +---------------------------------+
    |  ES regime stack + first-pass   |
    +---------------------------------+
    |  NQ regime stack + first-pass   |
    +---------------------------------+
    |  Footer / disclaimer            |
    +---------------------------------+

Used by `dashboard/build_static.py` and exposed as a CLI:

    python -m dashboard.brochure                    # default 30d horizon
    python -m dashboard.brochure --days 60
    python -m dashboard.brochure --out path.png
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Make sibling project modules importable when invoked as `python -m dashboard.brochure`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from macro.hmm_regime import HMMRegimeDetector, REGIME_LABELS, DEFAULT_PRIMARIES  # noqa: E402

# --- Brand palette (matches the dashboard CSS) ---
BG_COLOR = "#0f1216"
PANEL_COLOR = "#161b22"
TEXT_PRIMARY = "#e7eaee"
TEXT_SECONDARY = "#8e98a6"
BORDER_COLOR = "#222932"

REGIME_PALETTE: dict[str, dict[str, str]] = {
    "BULL":   {"bg": "#1e3a26", "fg": "#6dd47e", "fill": "#2ecc71"},
    "NORMAL": {"bg": "#3a3416", "fg": "#f5d76e", "fill": "#f1c40f"},
    "BEAR":   {"bg": "#3d2718", "fg": "#f0a060", "fill": "#e67e22"},
    "CRASH":  {"bg": "#3a181a", "fg": "#ff6b6b", "fill": "#c0392b"},
}

DEFAULT_FIGSIZE = (8.5, 11.0)  # portrait, fits a phone screen well
DEFAULT_DPI = 140


# --------------------------------------------------------------------- helpers


def _collect_asset_data(det: HMMRegimeDetector, n_days: int) -> dict[str, Any]:
    """Pull the forecast horizon DataFrame and key derived metrics for one asset."""
    df = det.forecast_horizon(n_days=n_days)
    fpt = det.expected_first_passage_to_dump()
    bars_per_day = max(1, det.bars_per_day)

    def _row_at(target_day: float) -> dict[str, float]:
        target_bar = int(round(target_day * bars_per_day))
        target_bar = min(target_bar, len(df) - 1)
        return df.iloc[target_bar].to_dict()

    current = _row_at(0.0)
    one_day = _row_at(1.0)
    two_day = _row_at(2.0)

    regime_columns = [f"p_{lab}" for lab in REGIME_LABELS[: det.n_states] if f"p_{lab}" in df.columns]
    current_regime_probs = {col[2:]: float(current[col]) for col in regime_columns}
    current_regime = max(current_regime_probs.items(), key=lambda kv: kv[1])[0]

    return {
        "asset": det.primary_asset,
        "df": df,
        "fpt_days": fpt,
        "current_regime": current_regime,
        "p_dump_now": float(current["p_dump"]),
        "p_dump_1d": float(one_day["p_dump"]),
        "p_dump_2d": float(two_day["p_dump"]),
        "expected_2d_return": float(two_day["cum_e_logret"]),
    }


def _draw_title(fig: plt.Figure, ax: plt.Axes, title: str, n_days: int) -> None:
    ax.set_facecolor(BG_COLOR)
    ax.text(
        0.5, 0.65, title,
        ha="center", va="center",
        color=TEXT_PRIMARY, fontsize=22, fontweight="bold",
        family="sans-serif",
    )
    subtitle = (
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        f"   |   horizon: {n_days} trading days"
    )
    ax.text(
        0.5, 0.20, subtitle,
        ha="center", va="center",
        color=TEXT_SECONDARY, fontsize=11,
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_status_card(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Per-asset card with current regime, dump probabilities, expected drift."""
    palette = REGIME_PALETTE.get(data["current_regime"], REGIME_PALETTE["NORMAL"])
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Card background with regime-colored border
    bbox = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=PANEL_COLOR, edgecolor=palette["fill"],
        linewidth=2.5, transform=ax.transAxes,
    )
    ax.add_patch(bbox)

    # Asset symbol
    ax.text(
        0.5, 0.86, data["asset"],
        ha="center", va="center", transform=ax.transAxes,
        color=TEXT_PRIMARY, fontsize=20, fontweight="bold",
    )

    # Regime pill
    pill = FancyBboxPatch(
        (0.18, 0.62), 0.64, 0.14,
        boxstyle="round,pad=0,rounding_size=0.07",
        facecolor=palette["bg"], edgecolor="none",
        transform=ax.transAxes,
    )
    ax.add_patch(pill)
    ax.text(
        0.5, 0.69, data["current_regime"],
        ha="center", va="center", transform=ax.transAxes,
        color=palette["fg"], fontsize=14, fontweight="bold",
    )

    # 4-up stats row
    stats = [
        ("P(now)",  f"{data['p_dump_now'] * 100:.0f}%"),
        ("P(1d)",   f"{data['p_dump_1d'] * 100:.0f}%"),
        ("P(2d)",   f"{data['p_dump_2d'] * 100:.0f}%"),
        ("E[2d]",   f"{data['expected_2d_return'] * 100:+.1f}%"),
    ]
    for k, (label, value) in enumerate(stats):
        x = 0.125 + (k * 0.25)
        ax.text(
            x, 0.42, label,
            ha="center", va="center", transform=ax.transAxes,
            color=TEXT_SECONDARY, fontsize=9,
        )
        ax.text(
            x, 0.27, value,
            ha="center", va="center", transform=ax.transAxes,
            color=TEXT_PRIMARY, fontsize=14, fontweight="bold",
        )

    # First-passage line
    fpt = data["fpt_days"]
    if fpt is None:
        fpt_text = "first dump: unreachable"
    elif fpt == 0.0:
        fpt_text = "first dump: now"
    else:
        fpt_text = f"first dump: ~{fpt:.1f}d"
    ax.text(
        0.5, 0.10, fpt_text,
        ha="center", va="center", transform=ax.transAxes,
        color=TEXT_SECONDARY, fontsize=10,
    )


def _draw_regime_chart(ax: plt.Axes, data: dict[str, Any], show_legend: bool, show_xlabel: bool) -> None:
    """Stacked area of forward regime probabilities for one asset."""
    df = data["df"]
    days = df["day"].values

    ax.set_facecolor(PANEL_COLOR)
    ordered = [lab for lab in ["BULL", "NORMAL", "BEAR", "CRASH"] if f"p_{lab}" in df.columns]
    stacks = [df[f"p_{lab}"].values for lab in ordered]
    colors = [REGIME_PALETTE[lab]["fill"] for lab in ordered]
    ax.stackplot(days, *stacks, labels=ordered, colors=colors, alpha=0.92)

    ax.set_xlim(0, max(1.0, float(days.max())))
    ax.set_ylim(0, 1)
    ax.set_ylabel("regime", color=TEXT_SECONDARY, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("trading days from now", color=TEXT_SECONDARY, fontsize=10)

    # Inline asset label
    ax.text(
        0.015, 0.92, data["asset"],
        transform=ax.transAxes, color=TEXT_PRIMARY,
        fontsize=14, fontweight="bold",
        bbox=dict(facecolor=BG_COLOR, edgecolor="none", pad=4, alpha=0.85),
    )

    # First-passage marker
    fpt = data["fpt_days"]
    if fpt is not None and 0.0 < fpt <= float(days.max()):
        ax.axvline(fpt, color="#ffffff", linestyle=":", linewidth=1.5, alpha=0.75)
        ax.text(
            fpt, 0.05, f" {fpt:.1f}d ",
            color="#ffffff", fontsize=8, va="bottom", ha="left",
            bbox=dict(facecolor=BG_COLOR, edgecolor="#ffffff", pad=1.5, linewidth=0.5),
        )

    if show_legend:
        leg = ax.legend(
            loc="upper right", framealpha=0.85, ncol=4, fontsize=8,
            facecolor=BG_COLOR, edgecolor="none",
        )
        for text in leg.get_texts():
            text.set_color(TEXT_PRIMARY)

    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    ax.grid(True, alpha=0.18, color=TEXT_SECONDARY, linestyle="-", linewidth=0.4)


def _draw_footer(ax: plt.Axes) -> None:
    ax.set_facecolor(BG_COLOR)
    ax.text(
        0.5, 0.55,
        "Conditional regime similarity scores from a fitted Gaussian HMM.",
        ha="center", va="center",
        color=TEXT_SECONDARY, fontsize=9, style="italic",
    )
    ax.text(
        0.5, 0.20,
        "Treat as a risk dial, not a directional crash call.",
        ha="center", va="center",
        color=TEXT_SECONDARY, fontsize=9, style="italic",
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ----------------------------------------------------------------- public API


def make_brochure(
    detectors: list[HMMRegimeDetector],
    output_path: str | Path,
    n_days: int = 30,
    title: str = "ARIA HMM Regime Forecast",
) -> Path:
    """Render and save the brochure PNG. Returns the saved path.

    `detectors` should be a list of *already-fit* HMMRegimeDetector instances
    (call `.fit()` or `.load()` first). Returns the output path.
    """
    if not detectors:
        raise ValueError("make_brochure needs at least one detector")

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    asset_data = [_collect_asset_data(d, n_days) for d in detectors]
    n_assets = len(asset_data)

    fig = plt.figure(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, facecolor=BG_COLOR)

    # 1 title row + 1 cards row + n_assets chart rows + 1 footer row
    height_ratios = [0.7, 1.4] + [2.0] * n_assets + [0.4]
    n_rows = len(height_ratios)
    gs = gridspec.GridSpec(
        nrows=n_rows,
        ncols=n_assets,
        height_ratios=height_ratios,
        hspace=0.42,
        wspace=0.18,
        left=0.07, right=0.93, top=0.96, bottom=0.04,
    )

    # Title (spans all columns)
    ax_title = fig.add_subplot(gs[0, :])
    _draw_title(fig, ax_title, title, n_days)

    # Status cards (one column per asset)
    for i, d in enumerate(asset_data):
        ax_card = fig.add_subplot(gs[1, i])
        _draw_status_card(ax_card, d)

    # Per-asset regime charts (each spans all columns)
    for i, d in enumerate(asset_data):
        ax_chart = fig.add_subplot(gs[2 + i, :])
        _draw_regime_chart(
            ax_chart,
            d,
            show_legend=(i == 0),
            show_xlabel=(i == n_assets - 1),
        )

    # Footer (spans all columns)
    ax_footer = fig.add_subplot(gs[-1, :])
    _draw_footer(ax_footer)

    fig.savefig(output_path, dpi=DEFAULT_DPI, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ------------------------------------------------------------------------- CLI


def _parse_args(argv: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "out_path": Path("dashboard/site/brochure.png"),
        "assets": list(DEFAULT_PRIMARIES),
        "days": 30,
    }
    i = 1
    while i < len(argv):
        tok = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if tok == "--out" and nxt is not None:
            out["out_path"] = Path(nxt); i += 2; continue
        if tok == "--asset" and nxt is not None:
            out["assets"] = [a.strip().upper() for a in nxt.split(",") if a.strip()]
            i += 2; continue
        if tok == "--days" and nxt is not None:
            out["days"] = int(nxt); i += 2; continue
        if tok in ("-h", "--help"):
            print(__doc__); sys.exit(0)
        i += 1
    return out


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    detectors: list[HMMRegimeDetector] = []
    for asset in args["assets"]:
        det = HMMRegimeDetector(primary_asset=asset)
        if not det.load():
            det.fit()
        detectors.append(det)
    path = make_brochure(detectors, args["out_path"], n_days=args["days"])
    print(f"brochure saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
