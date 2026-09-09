"""Bootstrap: fetch everything, measure everything, train the models.

Run once to create the models, and again whenever the feature code changes. It
measures before it trains, and it prints what each feature group is worth,
because a number nobody looked at is not evidence.

    python build.py                    # all three horizons
    python build.py --horizons 21      # just one
    python build.py --sample 150       # a subset of the universe, for a dry run
    python build.py --no-earnings      # skip the slow per-ticker fetch
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

import config
import earnings as earnings_mod
import features
import model as model_mod
import prices
import universe

log = logging.getLogger("build")

METRICS_PATH = config.DATA / "metrics.json"


def rule(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}", flush=True)


def shrink(df: pd.DataFrame) -> pd.DataFrame:
    """Halve the memory. A thousand tickers over twenty years is millions of
    rows, and float64 throughout will not fit a standard runner."""
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


def assemble(sample: int | None, want_earnings: bool,
             earnings_budget: float) -> tuple[pd.DataFrame, dict]:
    rule("UNIVERSE")
    uni = universe.Universe.load()
    s = uni.summary()
    print(f"  history source        {uni.source}")
    print(f"  in the index today    {s['current']}")
    print(f"  ever a member         {s['ever_in_window']}")
    print(f"  came and went         {s['turnover']}")
    if s["survivorship_safe"]:
        cc = uni.count_check()
        print(f"  membership count      {cc['min']}-{cc['max']} "
              f"(median {cc['median']}) {'OK' if cc['ok'] else 'SUSPECT'}")
    else:
        print("  !! SURVIVORSHIP-BIASED: no membership history. Volatility "
              "will be understated.")

    tickers = uni.all_ever()
    if sample and sample < len(tickers):
        step = len(tickers) / sample
        tickers = [tickers[int(i * step)] for i in range(sample)]
        print(f"  sampling {len(tickers)} of them for this run")

    rule("PRICES")
    t0 = time.time()
    px = prices.load(tickers)
    print(f"  {len(px):,} rows, {px['ticker'].nunique()} tickers, "
          f"{px['date'].min().date()} to {px['date'].max().date()} "
          f"in {time.time() - t0:.0f}s")
    a = prices.audit(prices.add_returns(px))
    print(f"  adjustment: {a['implausible_moves']} impossible moves "
          f"({a['implausible_share'] * 100:.3f}%), "
          f"{a['split_shaped_moves']} split-shaped")
    if a["implausible_share"] > 0.003:
        print("  !! too many impossible days - the volatility numbers below "
              "are suspect")

    market = None
    try:
        market = prices.load(config.BENCHMARK_TICKERS)
    except prices.PricesUnavailable as exc:
        print(f"  no market reference: {exc}")

    earn = None
    if want_earnings:
        rule("EARNINGS")
        t0 = time.time()
        earn = earnings_mod.load(tickers, budget_seconds=earnings_budget)
        cov = earnings_mod.coverage(earn, tickers)
        print(f"  {cov['dates']:,} dates for {cov['tickers_with_dates']} "
              f"tickers ({cov['share'] * 100:.0f}% of the universe) "
              f"in {time.time() - t0:.0f}s")
        if cov["share"] < 0.5:
            print("  thin coverage - expect the earnings group to measure as "
                  "a null for lack of data rather than lack of effect")

    rule("FEATURES")
    t0 = time.time()
    panel = shrink(features.build_panel(px, market))
    del px
    gc.collect()
    print(f"  {len(panel):,} ticker-days x {len(panel.columns)} columns "
          f"in {time.time() - t0:.0f}s")
    return panel, {"universe": s, "adjustment": a,
                   "earnings": earnings_mod.coverage(earn, tickers) if earn is not None else None,
                   "tickers": len(tickers)}, earn


def run_horizon(panel: pd.DataFrame, label: str, horizon: int,
                earn, do_ablation: bool) -> dict:
    rule(f"HORIZON: {label.upper()}  ({horizon} trading days)")
    df = shrink(features.build_for_horizon(panel, horizon, earn))
    X, carry = features.training_matrix(df)
    print(f"  {len(X):,} usable rows")

    out = {}
    if do_ablation:
        print("\n  What each feature group is worth. Each is measured ON ITS")
        print("  OWN against the volatility core, paired window by window,")
        print("  with the t statistic beside it. |t| under 2 means the")
        print("  difference cannot be told from zero.\n")
        abl = model_mod.ablation(df, horizon, features.FEATURE_GROUPS)
        out["ablation"] = abl
        if abl:
            print(f"  {'group':<10} {'cols':>5} {'pinball':>9} {'delta':>10} "
                  f"{'t':>7}  {'coverage':>9}   verdict")
            for name, r in abl.items():
                t = r.get("t")
                ts = "      -" if t is None or pd.isna(t) else f"{t:+7.2f}"
                print(f"  {name:<10} {r['columns']:>5} {r['pinball']:>9.4f} "
                      f"{r['delta']:>+10.5f} {ts}  "
                      f"{r['coverage'] * 100:8.1f}%   {r.get('verdict', '')}")

    print("\n  Walk-forward evaluation:")
    oos = model_mod.walk_forward(df, horizon)
    metrics = model_mod.evaluate(oos, horizon)
    out["metrics"] = metrics
    print(model_mod.summarize(metrics))

    print("\n  Fitting the final model on everything:")
    m = model_mod.RangeModel(horizon=horizon).fit(X, carry["z"])
    path = config.MODELS / f"range_h{horizon}.joblib"
    try:
        import joblib
        joblib.dump(m, path)
    except ImportError:
        import pickle
        path = path.with_suffix(".pkl")
        with open(path, "wb") as fh:
            pickle.dump(m, fh)
    print(f"  trained on {m.trained_rows:,} rows, saved to {path.name}")
    print(f"  width calibration: "
          f"{', '.join(f'{q:.2f}->{v:.2f}' for q, v in m.spread.items())}")

    del df, X, carry, oos
    gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", nargs="*", type=int)
    ap.add_argument("--sample", type=int)
    ap.add_argument("--no-earnings", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--earnings-budget", type=float, default=1800.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    panel, context, earn = assemble(args.sample, not args.no_earnings,
                                    args.earnings_budget)

    horizons = ({str(h): h for h in args.horizons} if args.horizons
                else dict(config.HORIZONS))
    results = {"context": context, "horizons": {}}
    for label, h in horizons.items():
        results["horizons"][label] = run_horizon(
            panel, label, h, earn, not args.no_ablation)

    model_mod.save_metrics(results, METRICS_PATH)

    rule("HONEST SUMMARY")
    print("The bar is not 50% coverage - the naive band already reaches that")
    print("overall. The bar is being calibrated for calm stocks AND wild ones")
    print("at the same time, without a wider band.\n")
    for label, r in results["horizons"].items():
        m = r.get("metrics") or {}
        if not m:
            continue
        print(f"  {label}")
        print(f"    pinball vs naive        {m['pinball_gain_pct']:+.2f}%")
        print(f"    coverage                {m['coverage'] * 100:.1f}% "
              f"(naive {m['coverage_naive'] * 100:.1f}%, target "
              f"{m['coverage_target'] * 100:.0f}%)")
        print(f"    worst volatility bucket {m['worst_bucket_error'] * 100:.1f} "
              f"points off (naive {m['worst_bucket_error_naive'] * 100:.1f})")
        print(f"    independent windows     {m['n_independent']:,}")
        if m["pinball_gain"] <= 0:
            print("    -> LOSES to the naive band. Either the panel is too "
                  "small or something is wrong; do not ship this horizon.")
    print(f"\nFull metrics written to {METRICS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
