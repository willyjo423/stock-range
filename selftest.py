"""One live run that says whether this project is worth building.

It does the usual schema reporting - what arrived, how much history, which
source answered - but the part that matters is the last section. Before writing
a model, it measures the premise the whole thing rests on:

    Does trailing volatility predict future volatility?

If that correlation comes back near zero, nothing downstream can work and the
honest thing is to stop. The sports builds each spent a day discovering their
premise was wrong (comparables predicted covers at exactly 50%) *after* the
model was built. This asks first.

    python selftest.py                # a sample of the index, quick
    python selftest.py --full         # every member, slow
    python selftest.py --sample 120
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

import config
import prices
import universe

log = logging.getLogger("selftest")

RULE = "=" * 76
THIN = "-" * 76


def head(t: str) -> None:
    print(f"\n{RULE}\n{t}\n{RULE}", flush=True)


def sub(t: str) -> None:
    print(f"\n{t}\n{THIN}", flush=True)


# ------------------------------------------------------------------ universe
def universe_block() -> universe.Universe | None:
    head("1. THE UNIVERSE")
    try:
        uni = universe.Universe.load()
    except universe.UniverseUnavailable as exc:
        print(f"  FAILED: {exc}")
        print("  Without membership there is nothing to model. Send me this.")
        return None

    s = uni.summary()
    print(f"  in the index today          {s['current']}")
    print(f"  membership change records   {s['change_dates']}")
    print(f"  ever a member since {config.HISTORY_START[:4]}    {s['ever_in_window']}")
    print(f"  names that came and went    {s['turnover']}")

    if s["survivorship_safe"]:
        print(f"\n  Survivorship handled. {s['turnover']} companies that left the")
        print("  index are still in the training set. Those are disproportionately")
        print("  the ones whose volatility exploded before they went - drop them")
        print("  and the model learns that stocks are calmer than they are.")
    else:
        print("\n  !! NO MEMBERSHIP HISTORY. Everything below is survivorship-")
        print("     biased and will understate volatility. Fixable, but I need")
        print("     to know before trusting any backtest.")
    return uni


# ------------------------------------------------------------------- prices
def price_block(tickers: list[str]) -> pd.DataFrame | None:
    head("2. PRICES")
    print(f"  requesting {len(tickers)} tickers from {config.HISTORY_START}")
    try:
        px = prices.load(tickers)
    except prices.PricesUnavailable as exc:
        print(f"  FAILED: {exc}")
        return None

    got = set(px["ticker"].unique())
    missing = [t for t in tickers if t not in got]
    print(f"  source that answered        {px['source'].iloc[0]}")
    print(f"  rows                        {len(px):,}")
    print(f"  tickers returned            {len(got)} of {len(tickers)}")
    print(f"  date range                  {px['date'].min().date()} to "
          f"{px['date'].max().date()}")
    if missing:
        print(f"  no data for                 {len(missing)}: "
              f"{', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")

    per = px.groupby("ticker").size()
    sub("HOW MUCH HISTORY, PER TICKER")
    for label, lo, hi in (("under 1 year", 0, 252),
                          ("1 to 3 years", 252, 756),
                          ("3 to 10 years", 756, 2520),
                          ("over 10 years", 2520, 10 ** 9)):
        n = int(((per >= lo) & (per < hi)).sum())
        print(f"  {label:<16} {n:>4} tickers")
    print(f"\n  median {int(per.median()):,} trading days "
          f"({per.median() / 252:.1f} years)")
    print(f"  usable at the {config.MIN_HISTORY_DAYS}-day minimum: "
          f"{int((per >= config.MIN_HISTORY_DAYS).sum())}")
    return px


def adjustment_block(px: pd.DataFrame) -> pd.DataFrame:
    head("3. IS THE DATA SPLIT-ADJUSTED?")
    print("An unadjusted 2-for-1 split reads as a 50% single-day loss. Feed")
    print("those to a volatility model and it learns that stocks are twice as")
    print("wild as they are - and nothing about that failure is loud.\n")

    df = prices.add_returns(px)
    a = prices.audit(df)
    print(f"  daily returns                    {a['returns']:,}")
    print(f"  daily standard deviation         {a['daily_sd'] * 100:.2f}%")
    print(f"  moves beyond "
          f"{config.IMPLAUSIBLE_DAILY_MOVE:.0%}                {a['implausible_moves']:,} "
          f"({a['implausible_share'] * 100:.3f}%)")
    print(f"  moves clustered near a 2-for-1   {a['split_shaped_moves']:,}")
    print(f"  largest single-day move          {a['max_abs_move'] * 100:.1f}%")

    print()
    if a["implausible_share"] < 0.0005:
        print("  VERDICT: adjusted. Rare extreme days are real news.")
    elif a["implausible_share"] < 0.003:
        print("  VERDICT: probably adjusted, with a tail worth a look. Send me")
        print("  this block.")
    else:
        print("  VERDICT: SUSPECT. Too many impossible days. Do not trust any")
        print("  volatility number until this is resolved.")
    return df


# ------------------------------------------------------------- the premise
def premise_block(df: pd.DataFrame) -> None:
    head("4. THE PREMISE: DOES VOLATILITY PERSIST?")
    print("Everything here rests on one claim - that how much a stock has been")
    print("moving tells you how much it is about to move. If this is weak, the")
    print("project does not work and it is better to find out now.\n")

    rv = prices.realized_vol(df)
    rows = []
    for label, h in config.HORIZONS.items():
        d = rv.copy()
        # Realized volatility over the NEXT h days, which is what a range is.
        fwd = (d.sort_values(["ticker", "date"]).groupby("ticker")["ret"]
                .rolling(h, min_periods=max(3, h // 2)).std()
                .reset_index(level=0, drop=True).shift(-h) * np.sqrt(252))
        d["fwd_vol"] = fwd
        # Compare like with like: trailing window of the same length.
        col = f"rv_{h}" if f"rv_{h}" in d.columns else "rv_21"
        ok = d[[col, "fwd_vol"]].dropna()
        if len(ok) < 500:
            continue
        r = float(np.corrcoef(ok[col], ok["fwd_vol"])[0, 1])
        rows.append((label, h, r, len(ok)))
        print(f"  {label:<9} trailing {col:<6} vs next {h:>2} days:  "
              f"corr {r:+.3f}   (n={len(ok):,})")

    if not rows:
        print("  not enough overlapping history to measure")
        return

    best = max(r for _, _, r, _ in rows)
    print()
    if best > 0.5:
        print("  VERDICT: STRONG. Volatility persists, which is the whole")
        print("  premise. Worth building.")
    elif best > 0.3:
        print("  VERDICT: REAL BUT MODEST. Worth building, with expectations set.")
    else:
        print("  VERDICT: WEAK. This is the number that decides whether to")
        print("  continue. Send it to me before we go further.")

    print("\n  NOTE: these correlations are computed over overlapping windows,")
    print("  so the sample is far smaller than n suggests and the figure is")
    print("  optimistic. It is a go/no-go signal, not a result.")


def baseline_block(df: pd.DataFrame) -> None:
    head("5. THE BAR: HOW GOOD IS THE OBVIOUS ANSWER?")
    print("The simplest possible range: assume the next month looks like the")
    print("last month. Any model has to beat this, and the sports builds are a")
    print("reminder of how often the obvious answer is hard to improve on.\n")

    rv = prices.realized_vol(df)
    for label, h in config.HORIZONS.items():
        d = rv.copy()
        d["fwd_ret"] = prices.forward_return(d, h)
        col = f"rv_{h}" if f"rv_{h}" in d.columns else "rv_21"
        ok = d[[col, "fwd_ret"]].dropna()
        if len(ok) < 500:
            continue
        # A middle-half band from the trailing volatility alone.
        halfwidth = 0.6745 * ok[col] * np.sqrt(h / 252.0)
        inside = float((ok["fwd_ret"].abs() <= halfwidth).mean())
        print(f"  {label:<9} naive band held {inside * 100:5.1f}% of the time "
              f"(target 50%)   n={len(ok):,}")
        if inside < 0.42:
            print(f"            -> too narrow: real returns have fatter tails "
                  f"than the normal assumption")
        elif inside > 0.58:
            print(f"            -> too wide")
    print("\n  A model earns its place by moving these toward 50% and by")
    print("  tightening the band on the stocks that deserve a tighter one.")


def benchmark_block() -> None:
    head("6. INDEX AND VOLATILITY REFERENCES")
    try:
        bm = prices.load(config.BENCHMARK_TICKERS, ttl=config.CACHE_TTL_SECONDS)
    except prices.PricesUnavailable as exc:
        print(f"  none available: {exc}")
        print("  Not fatal - these are context features, not the target.")
        return
    for t, block in bm.groupby("ticker"):
        print(f"  {t:<8} {len(block):>6,} days   "
              f"{block['date'].min().date()} to {block['date'].max().date()}")
    missing = [t for t in config.BENCHMARK_TICKERS
               if t not in set(bm["ticker"].unique())]
    if missing:
        print(f"\n  unavailable: {', '.join(missing)}")


def earnings_block(tickers: list[str]) -> None:
    head("7. EARNINGS DATES")
    print("A window containing an earnings release is a different animal from")
    print("one that does not. This checks whether the dates are reachable and,")
    print("more importantly, how far back they go - a feature that only exists")
    print("for the last two years cannot be trained on twenty.\n")
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed; cannot check")
        return

    found, spans = 0, []
    for t in tickers[:8]:
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=40)
        except Exception as exc:  # noqa: BLE001 - probing, not relying
            print(f"  {t:<6} unavailable ({type(exc).__name__})")
            continue
        if df is None or len(df) == 0:
            print(f"  {t:<6} no dates returned")
            continue
        idx = pd.to_datetime(pd.Series(df.index), errors="coerce", utc=True).dropna()
        if idx.empty:
            continue
        found += 1
        years = (idx.max() - idx.min()).days / 365.25
        spans.append(years)
        print(f"  {t:<6} {len(idx):>3} dates, {idx.min().date()} to "
              f"{idx.max().date()}  ({years:.1f} years)")

    print()
    if not found:
        print("  VERDICT: unavailable. The earnings feature gets dropped and")
        print("  the model leans on volatility alone - workable, but it will")
        print("  under-widen the windows that straddle a report.")
    elif np.median(spans) < 5:
        print(f"  VERDICT: present but shallow (median {np.median(spans):.1f} "
              f"years). Usable for live forecasts, not for training on two")
        print("  decades. I would treat it as a display note, not a feature.")
    else:
        print(f"  VERDICT: deep enough to train on (median "
              f"{np.median(spans):.1f} years).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=60,
                    help="how many tickers to probe (default 60)")
    ap.add_argument("--full", action="store_true", help="probe every member")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    head("RANGE FORECAST - DATA PROBE")
    print("Nothing here is fatal. A missing field is a finding; the run")
    print("finishes and prints the summary either way.")

    uni = universe_block()
    if uni is None:
        return 1

    members = uni.all_ever() if args.full else uni.current
    if not args.full and args.sample < len(members):
        # A deterministic spread across the alphabet rather than the first N,
        # which would be all A-names and not representative of anything.
        step = len(members) / args.sample
        members = [members[int(i * step)] for i in range(args.sample)]
        print(f"\n  probing a {len(members)}-ticker sample; --full for all")

    px = price_block(members)
    if px is None:
        return 1

    df = adjustment_block(px)
    premise_block(df)
    baseline_block(df)
    benchmark_block()
    earnings_block(members)

    head("WHAT TO SEND BACK")
    print("All of it. The three that decide what gets built next are section 1")
    print("(survivorship), section 4 (does volatility persist), and section 5")
    print("(how good the obvious answer already is).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
