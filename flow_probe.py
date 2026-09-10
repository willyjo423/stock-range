"""What the free option chains actually contain. Run this before believing any
of the flow code.

Every schema assumption in `chains.py` was written without a live endpoint in
front of it, which is exactly the situation that produced four separate bugs in
the football builds. So nothing downstream should be trusted until this has run
and its output has been read.

Five questions, in the order that decides whether to continue:

1. **Does either source answer at all**, and how fast? Five hundred names have
   to fit inside a scheduled run.
2. **Are the fields there** - volume, open interest, a two-sided quote - and do
   the contract symbols agree with the stated strikes and expiries?
3. **How many contracts survive each filter** on an ordinary day? If the answer
   is three thousand the filters are too loose to mean anything; if it is zero
   they are too tight to ever fire. Both are findable now rather than in a
   month.
4. **Does open interest hold still during the session?** The whole
   new-positioning filter rests on it. Needs two runs an hour apart, and the
   probe says so rather than guessing.
5. **What does a day of this cost** in requests and minutes.

    python flow_probe.py --sample 40
    python flow_probe.py --tickers AAPL NVDA TSLA
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

import chains
import config
import flow
import universe

log = logging.getLogger(__name__)


def head(t):
    print(f"\n{t}\n{'=' * 70}")


def sub(t):
    print(f"\n{t}\n{'-' * 70}")


# ------------------------------------------------------------------ 1 & 2
def probe_sources(names: list[str]) -> tuple[pd.DataFrame, dict]:
    head("1. WHICH SOURCE ANSWERS, AND HOW FAST")
    timings = {}
    frames = []
    errors = {}

    for source in ("cboe", "yfinance"):
        sample = names[:4] if source == "yfinance" else names[:8]
        t0 = time.time()
        got = 0
        for t in sample:
            try:
                df = (chains.from_cboe(t) if source == "cboe"
                      else chains.from_yfinance(t))
                if not df.empty:
                    got += 1
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001
                errors.setdefault(source, []).append(f"{t}: {str(exc)[:120]}")
        elapsed = time.time() - t0
        per = elapsed / max(len(sample), 1)
        timings[source] = {"answered": got, "tried": len(sample),
                           "seconds_each": round(per, 2)}
        print(f"  {source:<10} {got}/{len(sample)} answered   "
              f"{per:5.2f}s per name   "
              f"-> {per * 500 / 60:5.1f} min for 500 names, serially")
        for e in (errors.get(source) or [])[:3]:
            print(f"             {e}")

    if not frames:
        print("\n  Neither source returned a chain. Nothing downstream can "
              "work.\n  This is the stop-here result, and it is better to "
              "have it now.")
        return pd.DataFrame(), timings

    # Both sources are asked about the same leading tickers, so the same
    # contract arrives twice. The first run counted those twice and inflated
    # every share in section 3 - a filter that passes 20% of contracts looks
    # identical whether the contracts are distinct or duplicated. CBOE is
    # appended first, so keeping the first occurrence keeps the faster source.
    chain = pd.concat(frames, ignore_index=True)
    before = len(chain)
    chain = chain.drop_duplicates(["ticker", "contract"], keep="first")
    if len(chain) < before:
        print(f"\n  {before - len(chain):,} duplicate contracts dropped "
              f"(the two sources were asked about overlapping tickers)")

    head("2. WHAT THE CHAIN CONTAINS")
    q = chains.check(chain)
    print(f"  rows                    {q['rows']:,}")
    print(f"  tickers                 {q['tickers']}")
    print(f"  sources                 {', '.join(q['sources'])}")
    print(f"  distinct expiries       {q['expiries']}")
    print(f"  open interest missing   {q['oi_missing'] * 100:5.1f}%")
    print(f"  volume missing          {q['volume_missing'] * 100:5.1f}%")
    print(f"  spot missing            {q['spot_missing'] * 100:5.1f}%")
    print(f"  contracts with no trades {q['zero_volume_share'] * 100:5.1f}%"
          "   (high is normal and expected)")
    print(f"  symbol disagreements    {q['symbol_disagreements']}"
          "   (anything but 0 means the parser is wrong)")
    print(f"\n  verdict: {'USABLE' if q['usable'] else 'NOT USABLE'}")
    if not q["usable"]:
        print("  The fields this rests on are not there. Read the rows above "
              "before\n  changing any code - the fix is in chains.py's key "
              "names, not in the filters.")
    return chain, timings


# ---------------------------------------------------------------------- 3
def probe_gates(chain: pd.DataFrame) -> None:
    head("3. HOW MANY CONTRACTS SURVIVE EACH FILTER")
    if chain.empty:
        print("  no chain to screen")
        return
    rich = flow.enrich(chain)
    stepped = flow.interval(rich, None)
    screened = flow.screen(stepped)

    total = len(screened)
    print(f"  contracts read          {total:,}\n")
    print(f"  {'filter':<20} {'passes':>9} {'share':>8}")
    order = ["dte", "atm", "quote", "premium", "low_oi", "new_positioning"]
    for g in order:
        col = f"gate_{g}"
        if col not in screened:
            continue
        n = int(screened[col].sum())
        print(f"  {g:<20} {n:>9,} {n / max(total, 1) * 100:7.2f}%")

    n_flag = int(screened["flagged"].sum())
    tickers = screened[screened["flagged"] == 1]["ticker"].nunique()
    print(f"\n  all filters together    {n_flag:>9,} "
          f"{n_flag / max(total, 1) * 100:7.2f}%")
    print(f"  distinct names flagged  {tickers:>9}   "
          f"out of {chain['ticker'].nunique()} scanned")

    per_name = tickers / max(chain["ticker"].nunique(), 1)
    print(f"\n  Scaled to the full index that is roughly "
          f"{per_name * 500:.0f} names a day.")
    if per_name * 500 > 60:
        print("  That is too many to be a signal. Section 3b solves for the "
              "premium\n  floor that would fix it, rather than guessing at "
              "one.")
    elif per_name * 500 < 1:
        print("  That is too few to ever grade. Loosen one gate - but note "
              "which one,\n  because a filter set tuned until it fires is a "
              "filter set tuned on nothing.")
    else:
        print("  That is a workable rate: enough to accumulate a gradeable "
              "sample\n  within a couple of months, few enough to read.")

    # What the ordinary day looks like, so the thresholds are set against the
    # actual distribution rather than against a round number.
    near = screened[(screened["gate_dte"] == 1) & (screened["gate_atm"] == 1)
                    & (screened["volume"] > 0)]
    if len(near) > 20:
        sub("Premium among contracts that are short-dated, at the money, "
            "and traded")
        prem = near["day_premium"].dropna()
        for p in (50, 75, 90, 95, 99):
            print(f"  {p}th percentile   ${np.percentile(prem, p):>12,.0f}")
        print(f"  the $50,000 floor sits at the "
              f"{(prem < config.FLOW_MIN_PREMIUM).mean() * 100:.1f}th "
              f"percentile")
        vo = near["vol_oi"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vo) > 20:
            print(f"\n  volume/open-interest above 1.0: "
                  f"{(vo >= 1).mean() * 100:.1f}% of them")


def probe_threshold(chain: pd.DataFrame, target: int = 25) -> None:
    """Solve for the premium floor rather than guessing at it.

    The first live run measured that $50,000 sits at roughly the median of
    short-dated at-the-money contracts that traded at all. That is not a
    surprise once stated plainly: $50k was a criterion for one trade's premium,
    and what free data reports is a whole contract-day's premium. The number
    was right for a quantity this cannot see.

    So rather than pick a new number and defend it, this prints what each
    candidate floor would actually do to the daily count and lets the
    arithmetic choose. The target is deliberately modest - a couple of dozen
    names a day accumulates a gradeable sample within a few months and is still
    short enough to read.
    """
    head(f"3b. WHAT PREMIUM FLOOR WOULD MAKE THIS SELECTIVE (target ~{target}"
         f" names a day)")
    if chain.empty:
        print("  no chain to calibrate against")
        return

    screened = flow.screen(flow.interval(flow.enrich(chain), None))
    others = [c for c in screened.columns
              if c.startswith("gate_") and c != "gate_premium"]
    if not others:
        return
    base = screened[np.logical_and.reduce(
        [(screened[c] == 1).to_numpy() for c in others])]
    n_names = int(chain["ticker"].nunique())
    print(f"  contracts passing every filter except premium: {len(base):,}")
    print(f"  across {n_names} names scanned\n")
    if base.empty:
        print("  Nothing survives the other filters, so the premium floor is "
              "not what is\n  binding. Look at the moneyness band and the "
              "open-interest cap instead.")
        return

    print(f"  {'floor':>12} {'contracts':>10} {'names':>7} "
          f"{'projected/day':>14}")
    chosen = None
    for floor in (50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000,
                  5_000_000, 10_000_000):
        hit = base[base["new_premium"] >= floor]
        names = int(hit["ticker"].nunique())
        proj = names / max(n_names, 1) * 500
        print(f"  ${floor:>11,} {len(hit):>10,} {names:>7} {proj:>13.0f}")
        if chosen is None and proj <= target:
            chosen = (floor, proj)

    print()
    if chosen:
        print(f"  The lowest floor that lands near the target is "
              f"${chosen[0]:,}\n  (about {chosen[1]:.0f} names a day). Set "
              f"FLOW_MIN_PREMIUM to that.")
    else:
        print("  Even the highest floor here flags more than the target. The "
              "binding\n  constraint is not premium - tighten FLOW_ATM_BAND "
              "or FLOW_MAX_OI.")

    # The projection is a share measured on a handful of names, and a share
    # measured on a handful of names has enormous error bars. Saying so is the
    # difference between calibrating and fooling yourself.
    if n_names < 40:
        share = 0.5
        se = (share * (1 - share) / n_names) ** 0.5
        print(f"\n  CAUTION: {n_names} names is far too few to set a threshold "
              f"on. A rate\n  measured here carries a standard error of about "
              f"{se * 500:.0f} names a day, so\n  every projection above could "
              f"easily be double or half. Re-run with\n  --sample 80 during "
              f"market hours before changing anything.")


# ---------------------------------------------------------------------- 4
def probe_oi_stability(chain: pd.DataFrame) -> None:
    head("4. DOES OPEN INTEREST HOLD STILL DURING THE SESSION")
    print("  The new-positioning filter compares today's volume against open")
    print("  interest, and that only means anything if the open interest is")
    print("  yesterday's settled figure rather than something that ticks.")
    print()
    day = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    earlier = flow.load_snapshots(day)
    if not earlier:
        print("  No earlier snapshot from today, so this cannot be answered in")
        print("  a single run. Run the probe twice, at least an hour apart,")
        print("  during market hours - the second run answers it.")
        return

    prev = earlier[-1][1]
    key = ["ticker", "contract"]
    merged = (flow.enrich(chain)[key + ["open_interest", "volume"]]
              .merge(prev[key + ["open_interest", "volume"]], on=key,
                     suffixes=("_now", "_then")))
    if merged.empty:
        print("  No overlapping contracts with the earlier snapshot.")
        return
    changed = (merged["open_interest_now"] !=
               merged["open_interest_then"]).mean()
    traded = (merged["volume_now"] > merged["volume_then"]).mean()
    print(f"  contracts whose volume grew since the last snapshot: "
          f"{traded * 100:5.1f}%")
    print(f"  contracts whose open interest changed:               "
          f"{changed * 100:5.1f}%")
    if changed < 0.02:
        print("\n  Open interest held still while volume moved. That is the")
        print("  assumption the filter needs, and it holds.")
    else:
        print("\n  Open interest is moving intraday. The vol/OI ratio is then")
        print("  partly comparing today's trading against itself, and")
        print("  FLOW_MIN_VOL_OI is measuring less than it appears to.")


# ---------------------------------------------------------------------- 5
def probe_cost(timings: dict, n_names: int) -> None:
    head("5. WHAT A DAY OF THIS COSTS")
    best = min((v["seconds_each"] for v in timings.values()
                if v["answered"]), default=None)
    if best is None:
        print("  nothing answered, so there is nothing to cost")
        return
    workers = config.FLOW_MAX_WORKERS
    serial = best * n_names / 60
    parallel = serial / workers
    print(f"  fastest source          {best:.2f}s per name")
    print(f"  {n_names} names, serially      {serial:5.1f} min")
    print(f"  at {workers} in parallel        {parallel:5.1f} min per scan")
    print(f"  {config.FLOW_SCANS_PER_DAY} scans a day          "
          f"{parallel * config.FLOW_SCANS_PER_DAY:5.1f} min of runtime")
    print(f"  requests per day        "
          f"~{n_names * config.FLOW_SCANS_PER_DAY:,}")
    if parallel > 25:
        print("\n  That will not fit comfortably in a scheduled run. Either")
        print("  raise FLOW_MAX_WORKERS, or scan fewer names, or accept two")
        print("  scans a day instead of three.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=int, default=12)
    p.add_argument("--tickers", nargs="*")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(message)s")

    print("OPTIONS FLOW PROBE")
    print(f"run at {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC")
    print("US options trade 13:30-20:00 UTC; outside that, volume is "
          "yesterday's\nand section 3 describes a finished day rather than a "
          "live one.")

    if args.tickers:
        names = [universe.normalise_ticker(t) for t in args.tickers]
    else:
        try:
            names = universe.Universe.load().current[:args.sample]
        except Exception as exc:  # noqa: BLE001
            print(f"\n  universe unavailable ({exc}); falling back to a "
                  f"hand-picked sample")
            names = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META", "AMZN",
                     "GOOGL", "JPM", "XOM", "KO", "PG"][:args.sample]

    chain, timings = probe_sources(names)
    probe_gates(chain)
    probe_threshold(chain)
    probe_oi_stability(chain)
    probe_cost(timings, 500)

    if not chain.empty:
        try:
            flow.save_snapshot(flow.enrich(chain))
            print("\n  (snapshot stored, so a second run can answer section 4)")
        except Exception as exc:  # noqa: BLE001
            print(f"\n  could not store snapshot: {exc}")

    print("\n" + "=" * 70)
    print("Send this output back before any of the flow code is believed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
