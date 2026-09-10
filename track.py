"""Grade published ranges against what actually happened.

The bootstrap's walk-forward is honest but it is still the model marking work
on data that existed when it was written. This grades only ranges that were
archived *before* their window opened, and only once the window has closed.
It is the one measurement nobody can accidentally cheat on.

The headline is a single number per horizon: what share of outcomes landed
inside the stated middle half. It should be 50%. Materially under means the
ranges are too narrow, which is the failure that matters.

    python track.py
    python track.py --write     # also writes docs/results.html
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import config
import prices

log = logging.getLogger(__name__)


def load_forecasts() -> pd.DataFrame:
    """Every archived range, one row per ticker per horizon."""
    rows = []
    for path in sorted(config.FORECASTS.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("skipping %s: %s", path.name, exc)
            continue
        asof = payload.get("asof")
        for rec in payload.get("tickers") or []:
            for label, h in (rec.get("horizons") or {}).items():
                rows.append({
                    "asof": pd.Timestamp(asof), "ticker": rec["ticker"],
                    "horizon": label, "days": h.get("days"),
                    "close": rec.get("close"),
                    "low": h.get("low"), "high": h.get("high"),
                    "wide_low": h.get("wide_low"), "wide_high": h.get("wide_high"),
                    "earnings_inside": h.get("earnings_inside"),
                    "trailing_vol": rec.get("trailing_vol"),
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).dropna(subset=["low", "high", "days"])
    # The same window forecast on several days would be counted repeatedly, so
    # only the first published range for each ticker-horizon survives.
    return (df.sort_values("asof")
              .drop_duplicates(["ticker", "horizon", "asof"])
              .reset_index(drop=True))


def grade(fc: pd.DataFrame) -> pd.DataFrame:
    """Attach the actual close at the end of each window, where it has passed."""
    if fc.empty:
        return fc
    names = sorted(fc["ticker"].unique())
    start = (fc["asof"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    px = prices.load(names, start=start, ttl=6 * 3600)
    px = px.sort_values(["ticker", "date"])

    out = []
    for tkr, block in fc.groupby("ticker", sort=False):
        hist = px[px["ticker"] == tkr][["date", "close"]].reset_index(drop=True)
        if hist.empty:
            continue
        dates = hist["date"].to_numpy()
        closes = hist["close"].to_numpy()
        for row in block.itertuples(index=False):
            i = int(np.searchsorted(dates, np.datetime64(row.asof), "left"))
            end = i + int(row.days)
            if end >= len(closes):
                continue        # the window has not closed yet
            out.append({**row._asdict(), "settled": float(closes[end]),
                        "settle_date": pd.Timestamp(dates[end])})
    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out)
    df["inside"] = (df["settled"] >= df["low"]) & (df["settled"] <= df["high"])
    df["inside_wide"] = ((df["settled"] >= df["wide_low"])
                         & (df["settled"] <= df["wide_high"]))
    df["above"] = df["settled"] > df["high"]
    df["below"] = df["settled"] < df["low"]
    df["width_pct"] = 100 * (df["high"] - df["low"]) / df["close"]
    return df


def summarise(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    out = {"n": int(len(df)),
           "first": str(df["asof"].min().date()),
           "last": str(df["settle_date"].max().date()),
           "horizons": {}}
    for label, block in df.groupby("horizon"):
        rec = {
            "n": int(len(block)),
            "coverage": float(block["inside"].mean()),
            "coverage_wide": float(block["inside_wide"].mean()),
            "above": float(block["above"].mean()),
            "below": float(block["below"].mean()),
            "median_width_pct": float(block["width_pct"].median()),
        }
        # Same breakdown the backtest uses, so the two are comparable.
        if block["trailing_vol"].notna().sum() > 40:
            try:
                b = block.copy()
                b["bucket"] = pd.qcut(b["trailing_vol"], 4,
                                      labels=["calmest", "calm", "active",
                                              "wildest"], duplicates="drop")
                rec["by_volatility"] = {
                    str(k): {"n": int(len(g)), "coverage": float(g["inside"].mean())}
                    for k, g in b.groupby("bucket", observed=True)}
            except ValueError:
                pass
        out["horizons"][label] = rec
    return out


def report(s: dict) -> str:
    if not s:
        return ("No graded ranges yet. They appear once an archived forecast's "
                "window has closed - a week after the first daily run for the "
                "one-week horizon, a quarter for the longest.")
    lines = [
        "FORWARD-ONLY RESULTS  (ranges published before their window opened)",
        "=" * 68,
        f"Ranges graded : {s['n']:,}",
        f"Covering      : {s['first']} to {s['last']}",
        "",
        f"{'horizon':<10} {'n':>7} {'inside':>8} {'target':>7} "
        f"{'below':>7} {'above':>7} {'width':>8}",
    ]
    for label, r in s["horizons"].items():
        lines.append(
            f"{label:<10} {r['n']:>7,} {r['coverage'] * 100:7.1f}% "
            f"{'50%':>7} {r['below'] * 100:6.1f}% {r['above'] * 100:6.1f}% "
            f"{r['median_width_pct']:7.1f}%")

    for label, r in s["horizons"].items():
        if r["n"] < 100:
            continue
        if r["coverage"] < 0.42:
            lines.append(f"\n  {label}: ranges are too narrow "
                         f"({r['coverage'] * 100:.0f}% vs 50%). The model is "
                         f"overconfident and the bands need widening.")
        elif r["coverage"] > 0.60:
            lines.append(f"\n  {label}: ranges are wider than they need to be "
                         f"({r['coverage'] * 100:.0f}% vs 50%).")
        gap = r["above"] - r["below"]
        if abs(gap) > 0.12:
            side = "above" if gap > 0 else "below"
            lines.append(f"  {label}: misses are lopsided, mostly {side} the "
                         f"range. A range that fails asymmetrically is "
                         f"describing the wrong distribution.")

    for label, r in s["horizons"].items():
        if not r.get("by_volatility"):
            continue
        lines += ["", f"{label}, by how volatile the stock already was:"]
        for k, v in r["by_volatility"].items():
            lines.append(f"  {k:<9} {v['coverage'] * 100:6.1f}%  (n={v['n']:,})")
    return "\n".join(lines)


def render_html(s: dict) -> str:
    from dashboard import CSS
    body = report(s).replace("&", "&amp;").replace("<", "&lt;")
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Forward-only results</title>"
            f"<style>{CSS}pre{{white-space:pre-wrap;font:12.5px/1.6 "
            "ui-monospace,SFMono-Regular,Menlo,monospace}}</style></head><body>"
            '<div class="wrap"><h1>Forward-only results</h1>'
            f'<p class="sub">Updated {when}. Every range here was published '
            f'before its window opened and graded after it closed.</p>'
            f'<div class="note"><pre>{body}</pre></div>'
            '<footer><a href="./">Back to today\'s ranges</a></footer>'
            "</div></body></html>")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    fc = load_forecasts()
    if fc.empty:
        print("No archived forecasts yet.")
        return 0
    s = summarise(grade(fc))
    print(report(s))

    if args.write:
        (config.DOCS / "results.html").write_text(render_html(s))
        (config.DATA / "results.json").write_text(json.dumps(s, indent=2,
                                                            default=str))
        print(f"\nwrote {config.DOCS / 'results.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
