"""The options page: the model's range against what options are charging.

    python options_view.py
    python options_view.py --tickers AAPL NVDA

Reads the forecasts the daily run already published, fetches the option chains,
and answers one question per stock per horizon: is the market pricing more or
less movement than the model expects. Then attaches a direction, assembled from
the three inputs that carry any directional information at all.

Runs after `daily.py`, because it needs that run's forecasts.
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
import implied

log = logging.getLogger(__name__)


def load_forecasts(path: str | None = None) -> dict:
    path = path or (config.DOCS / "forecasts.json")
    try:
        return json.loads(open(path).read())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"No forecasts to work from ({exc}). Run daily.py "
                         f"first - this page is built on top of that one.")


def load_todays_flow(asof: str) -> dict[str, dict]:
    """Whatever the flow scanner flagged today, keyed by ticker.

    Most names will not be in here, and that is the normal case rather than a
    gap - the scan flags a few dozen out of five hundred. A missing entry means
    no unusual positioning was detected, which is different from bearish.
    """
    flags = flow.load_flags(asof)
    return {r["ticker"]: r for r in (flags.get("tickers") or []) if r.get("ticker")}


def build(tickers: list[str] | None = None) -> dict:
    payload = load_forecasts()
    asof = payload.get("asof")
    records = payload.get("tickers") or []
    if tickers:
        wanted = {t.upper() for t in tickers}
        records = [r for r in records if r["ticker"] in wanted]
    if not records:
        raise SystemExit("no forecasts matched")

    names = [r["ticker"] for r in records]
    log.info("fetching chains for %d names", len(names))
    chain, errors = chains.fetch_many(names)
    rich = flow.enrich(chain)
    by_ticker = {t: g for t, g in rich.groupby("ticker")}
    flow_flags = load_todays_flow(asof)

    out = []
    for rec in records:
        tkr = rec["ticker"]
        spot = rec.get("close")
        sub = by_ticker.get(tkr)
        row = {"ticker": tkr, "close": spot, "horizons": {},
               "has_flow": tkr in flow_flags}

        for label, h in (rec.get("horizons") or {}).items():
            days = h.get("days")
            model_pct = implied.model_sigma_pct(h.get("low"), h.get("high"))
            entry = {
                "days": days,
                "low": h.get("low"), "high": h.get("high"),
                "earnings_inside": h.get("earnings_inside"),
                "model_move_pct": round(model_pct, 2) if model_pct else None,
            }
            if sub is not None and days:
                exp_chain, dte = implied.pick_expiry(sub, days)
                imp = implied.implied_sigma_pct(exp_chain, spot, days, dte)
                entry.update(implied.compare(model_pct, imp))
            row["horizons"][label] = entry

        row["lean"] = implied.lean(
            # The tilt and the skew live on the shortest horizon, which is the
            # one the direction was measured on.
            next(iter((rec.get("horizons") or {}).values()), {}),
            flow_flags.get(tkr))
        out.append(row)

    # Biggest disagreement first, over whichever horizon disagrees most. A name
    # with no usable chain sorts to the bottom rather than being dropped, so
    # coverage stays visible.
    def gap(r):
        vals = [abs(h.get("ratio", 1.0) - 1.0)
                for h in r["horizons"].values() if h.get("ratio")]
        return max(vals, default=-1.0)

    out.sort(key=lambda r: -gap(r))

    priced = sum(1 for r in out if gap(r) >= 0)
    return {
        "asof": asof,
        "generated_at": payload.get("generated_at"),
        "tickers": out,
        "coverage": {"forecast": len(records), "priced": priced,
                     "no_chain": len(errors)},
        "model_metrics": payload.get("model_metrics") or {},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--out", default=str(config.DOCS / "options.json"))
    p.add_argument("--html", default=str(config.DOCS / "options.html"))
    p.add_argument("--min-gap", type=float, default=0.0,
                   help="only show names whose pricing gap exceeds this")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    payload = build(args.tickers)

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    import options_dashboard
    with open(args.html, "w") as fh:
        fh.write(options_dashboard.render(payload, min_gap=args.min_gap))

    kept = implied.archive(payload)
    cov = payload["coverage"]
    print(f"{cov['priced']} of {cov['forecast']} names priced "
          f"({cov['no_chain']} had no chain)")
    print(f"JSON -> {args.out}")
    print(f"HTML -> {args.html}")
    if kept:
        print(f"archived -> {kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
