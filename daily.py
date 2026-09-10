"""The daily run: today's ranges for the whole index.

    python daily.py                 # every current member
    python daily.py --tickers AAPL MSFT
    python daily.py --no-earnings

It loads the trained models, pulls enough recent history to compute the
features, forecasts each horizon, and writes a JSON payload plus a standalone
page. It also archives every forecast, so `track.py` can grade it once the
window has actually closed - which is the only measurement nobody can fool.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
import earnings as earnings_mod
import features
import prices
import universe
from model import FeatureMismatchError

log = logging.getLogger(__name__)

# Two years is plenty for a 252-day trailing window plus warm-up, and far
# cheaper than refetching two decades every morning.
LOOKBACK_DAYS = 900


def load_models() -> dict:
    """Whichever horizons have been trained. A missing one is skipped."""
    out = {}
    for label, h in config.HORIZONS.items():
        for suffix in (".joblib", ".pkl"):
            path = config.MODELS / f"range_h{h}{suffix}"
            if not path.exists():
                continue
            try:
                if suffix == ".joblib":
                    import joblib
                    out[label] = (h, joblib.load(path))
                else:
                    import pickle
                    with open(path, "rb") as fh:
                        out[label] = (h, pickle.load(fh))
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("could not load %s: %s", path.name, exc)
    if not out:
        raise SystemExit(
            f"No trained models in {config.MODELS}. Run the Bootstrap "
            f"workflow first - it trains them and commits them to the repo.")
    log.info("models loaded: %s", ", ".join(out))
    return out


def run(tickers: list[str] | None = None, want_earnings: bool = True) -> dict:
    models = load_models()

    if tickers:
        names = [universe.normalise_ticker(t) for t in tickers]
    else:
        names = universe.Universe.load().current
    log.info("forecasting %d tickers", len(names))

    start = (pd.Timestamp.today() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    px = prices.load(names, start=start, ttl=6 * 3600)
    market = None
    try:
        market = prices.load(config.BENCHMARK_TICKERS, start=start, ttl=6 * 3600)
    except prices.PricesUnavailable as exc:
        log.warning("no market reference: %s", exc)

    earn = earnings_mod.load(names, budget_seconds=600) if want_earnings else None

    panel = features.build_panel(px, market)
    asof = panel["date"].max()
    log.info("as of %s", asof.date())

    records: dict[str, dict] = {}
    for label, (h, model) in models.items():
        df = features.build_for_horizon(panel, h, earn)
        # The latest row per ticker is the only one being forecast. It has no
        # forward return by construction, which is the point.
        latest = (df.sort_values(["ticker", "date"])
                    .groupby("ticker", as_index=False).tail(1))
        latest = latest[latest["date"] >= asof - pd.Timedelta(days=7)]
        latest = latest.dropna(subset=["scale", "close"])
        if latest.empty:
            log.warning("no usable rows for %s", label)
            continue

        try:
            out = model.predict(latest[model.features], latest["scale"],
                                latest["close"])
        except FeatureMismatchError as exc:
            print(f"\n::error::{exc}\n")
            raise

        for (_, row), (_, q) in zip(latest.iterrows(), out.iterrows()):
            rec = records.setdefault(row["ticker"], {
                "ticker": row["ticker"],
                "asof": row["date"].strftime("%Y-%m-%d"),
                "close": round(float(row["close"]), 2),
                "trailing_vol": _j(row.get("rv_21"), 3),
                "horizons": {},
            })
            rec["horizons"][label] = {
                "days": h,
                "low": _j(q["px_q25"], 2), "high": _j(q["px_q75"], 2),
                "wide_low": _j(q["px_q10"], 2), "wide_high": _j(q["px_q90"], 2),
                "mid": _j(q["px_q50"], 2),
                "pct_low": _j(100 * (q["px_q25"] / row["close"] - 1), 1),
                "pct_high": _j(100 * (q["px_q75"] / row["close"] - 1), 1),
                "earnings_inside": _j(row.get("earnings_in_window"), 0),
                "days_to_earnings": _j(row.get("days_to_earnings"), 0),
            }

    metrics = {}
    if (config.DATA / "metrics.json").exists():
        try:
            metrics = json.loads((config.DATA / "metrics.json").read_text())
        except json.JSONDecodeError:
            pass

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asof": asof.strftime("%Y-%m-%d"),
        "tickers": sorted(records.values(), key=lambda r: r["ticker"]),
        "model_metrics": metrics,
    }


def _j(v, digits=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return None
    return round(float(v), digits)


def archive(payload: dict) -> str | None:
    if not payload.get("tickers"):
        return None
    path = config.FORECASTS / f"{payload['asof']}.json"
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--no-earnings", action="store_true")
    p.add_argument("--out", default=str(config.DOCS / "forecasts.json"))
    p.add_argument("--html", default=str(config.DOCS / "index.html"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    payload = run(args.tickers, want_earnings=not args.no_earnings)

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    from dashboard import render
    with open(args.html, "w") as fh:
        fh.write(render(payload))

    kept = archive(payload)
    print(f"{len(payload['tickers'])} tickers, as of {payload['asof']}")
    print(f"JSON -> {args.out}")
    print(f"HTML -> {args.html}")
    if kept:
        print(f"archived -> {kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
