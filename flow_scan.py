"""Run one flow scan and publish it.

    python flow_scan.py                     # every current index member
    python flow_scan.py --tickers AAPL NVDA
    python flow_scan.py --confirm           # also settle yesterday's flags

Designed to be run several times a session. Each run differences its chain
against the previous run's snapshot, folds what it finds into the day's flag
file, and rewrites the page. Running it once a day still works and simply
loses the burst measurement, which is the one thing that distinguishes a block
from a dribble - so once a day is a worse product, not a broken one.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

import chains
import config
import flow
import universe

log = logging.getLogger(__name__)


def previous_trading_day(day: str) -> str:
    d = pd.Timestamp(day) - pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def confirm_yesterday(chain: pd.DataFrame, day: str) -> dict:
    """Settle the previous session's flags against this morning's open interest.

    Worth doing before anything else in the run, because it uses the chain that
    has just been fetched and costs nothing extra. It is the free stand-in for
    the trade-level open/close flag: open interest that rose overnight by about
    what traded means the position was opened and held.
    """
    prev = previous_trading_day(day)
    if not flow.flags_path(prev).exists():
        return {}
    out = flow.confirm_open_interest(prev, chain)
    if out.get("checked"):
        log.info("open interest confirmed %d of %d flags from %s",
                 out["confirmed"], out["checked"], prev)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--prefer", default="cboe", choices=["cboe", "yfinance"])
    p.add_argument("--confirm", action="store_true",
                   help="settle yesterday's flags against today's open interest")
    p.add_argument("--out", default=str(config.DOCS / "flow.json"))
    p.add_argument("--html", default=str(config.DOCS / "flow.html"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.tickers:
        names = [universe.normalise_ticker(t) for t in args.tickers]
    else:
        names = universe.Universe.load().current
    log.info("scanning %d names", len(names))

    chain, errors = chains.fetch_many(names, prefer=args.prefer)
    fresh = flow.scan(names, chain=chain)
    fresh["errors"] = dict(list(errors.items())[:20])
    fresh["coverage"]["failed"] = len(errors)

    day = fresh["asof"]
    if args.confirm:
        fresh["oi_confirmation"] = confirm_yesterday(chain, day)

    merged = flow.merge_flags(flow.load_flags(day), fresh)
    flow.save_flags(day, merged)
    flow.prune()

    with open(args.out, "w") as fh:
        json.dump(merged, fh, indent=2, default=str)

    import flow_dashboard
    with open(args.html, "w") as fh:
        fh.write(flow_dashboard.render(merged))

    cov = merged.get("coverage") or {}
    print(f"{len(merged.get('tickers') or [])} names flagged from "
          f"{cov.get('with_chain')} chains "
          f"({cov.get('failed')} failed, "
          f"{cov.get('contracts'):,} contracts read)")
    print(f"gates: {cov.get('gate_survivors')}")
    print(f"JSON -> {args.out}")
    print(f"HTML -> {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
