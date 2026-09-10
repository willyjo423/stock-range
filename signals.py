"""Two ways of asking "is this one worth looking at".

The ranges on the page are all drawn outward from today's close, so today's
price sits in the middle of every one of them by construction. Asking where a
stock sits inside its own bracket has the same answer for every name. Getting
something useful needs a second reference point, and there are two honest ones.

**Term structure** - how wide the near-term band is against the long-term one,
once both are annualised so they are comparable. A stock whose next week is
priced as far wilder than its next quarter is one the model thinks is about to
do something, usually because short-run volatility is spiking or an earnings
report lands inside the window. This works from a single day's forecast.

**Position in a range published earlier** - a month ago the model said this
stock would land between two numbers. It is now near the top of that. That is
the truest reading of "at the edge of its bracket", and it needs the archive,
so it appears as the daily runs accumulate rather than on day one.

Neither is a prediction of direction. The first says a stock is likely to move;
it does not say which way. The second says a stock already has.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

TRADING_DAYS = 252
# Below this share of the window elapsed there has not been enough time for a
# position to mean anything, and the percentile is mostly noise.
MIN_ELAPSED = 0.25
# How far from the middle counts as worth flagging.
EDGE_PERCENTILE = 85.0


# ------------------------------------------------------------- term structure
def _annualised_width(h: dict) -> float | None:
    """Band width scaled to a common yardstick.

    A one-week band is naturally narrower than a three-month one, so comparing
    them raw says nothing. Dividing by the square root of the horizon puts both
    on an annual footing, which is the only way "wider than usual" has meaning
    across horizons.
    """
    if not h or h.get("pct_low") is None or h.get("pct_high") is None:
        return None
    days = h.get("days")
    if not days:
        return None
    width = (h["pct_high"] - h["pct_low"]) / 100.0
    return width / math.sqrt(days / TRADING_DAYS)


def term_structure(rec: dict) -> dict:
    """Near-term expected turbulence against the longest horizon available."""
    hz = rec.get("horizons") or {}
    if len(hz) < 2:
        return {}
    by_days = sorted(((v.get("days") or 0, k, v) for k, v in hz.items()))
    short_days, short_label, short = by_days[0]
    long_days, long_label, long = by_days[-1]

    a_short, a_long = _annualised_width(short), _annualised_width(long)
    if not a_short or not a_long or a_long <= 0:
        return {}

    ratio = a_short / a_long
    out = {
        "ratio": round(float(ratio), 3),
        "short": short_label, "long": long_label,
        "short_annual_pct": round(100 * a_short, 1),
        "long_annual_pct": round(100 * a_long, 1),
        "earnings_soon": int(short.get("earnings_inside") == 1),
    }
    # A ratio of 1 means the model sees the next week as no different from the
    # next quarter. Distance from 1 in either direction is the interesting part
    # - unusually calm is a state too, and often precedes the opposite.
    out["stress"] = round(float(abs(ratio - 1.0)), 3)
    out["direction"] = ("turbulent" if ratio > 1.05 else
                        "unusually calm" if ratio < 0.95 else "normal")
    return out


# --------------------------------------------------- position in open ranges
def _percentile_in(quantiles: list[tuple[float, float]], price: float) -> float:
    """Where a price falls in a distribution given by a few of its quantiles.

    Interpolated in log space, because the quantiles were built from log
    returns and a linear reading would bias the wide end. Outside the known
    quantiles it extrapolates gently and clamps, since the difference between
    the 97th and 99th percentile is not something five quantiles can resolve.
    """
    pts = sorted((q, p) for q, p in quantiles if p and p > 0)
    if len(pts) < 2 or price <= 0:
        return float("nan")
    x = math.log(price)
    qs = [q for q, _ in pts]
    ls = [math.log(p) for _, p in pts]

    if x <= ls[0]:
        span = ls[1] - ls[0]
        if span <= 0:
            return qs[0] * 100
        step = (qs[1] - qs[0]) * (x - ls[0]) / span
        return float(max(1.0, (qs[0] + step) * 100))
    if x >= ls[-1]:
        span = ls[-1] - ls[-2]
        if span <= 0:
            return qs[-1] * 100
        step = (qs[-1] - qs[-2]) * (x - ls[-1]) / span
        return float(min(99.0, (qs[-1] + step) * 100))
    return float(np.interp(x, ls, qs) * 100)


def load_archive(folder: Path | None = None) -> list[dict]:
    """Every archived daily forecast, oldest first."""
    folder = folder or config.FORECASTS
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("skipping %s: %s", path.name, exc)
    return out


def open_ranges(records: list[dict], asof: str,
                archive: list[dict] | None = None) -> dict:
    """For each ticker, where today sits inside ranges still running.

    Only windows that are open and at least a quarter elapsed are considered,
    and for each horizon the one furthest through its window wins - that is the
    one with the most evidence behind its position.
    """
    archive = load_archive() if archive is None else archive
    today = pd.Timestamp(asof)
    prices_now = {r["ticker"]: r.get("close") for r in records}

    best: dict[tuple[str, str], dict] = {}
    for payload in archive:
        published = pd.Timestamp(payload.get("asof", ""))
        if pd.isna(published) or published >= today:
            continue
        for rec in payload.get("tickers") or []:
            tkr = rec.get("ticker")
            price = prices_now.get(tkr)
            if price is None:
                continue
            for label, h in (rec.get("horizons") or {}).items():
                days = h.get("days")
                if not days:
                    continue
                # Trading days, approximately, from calendar days elapsed.
                elapsed_cal = (today - published).days
                elapsed = elapsed_cal * 252 / 365.25
                frac = elapsed / days
                if frac < MIN_ELAPSED or frac >= 1.0:
                    continue
                pct = _percentile_in(
                    [(0.10, h.get("wide_low")), (0.25, h.get("low")),
                     (0.50, h.get("mid")), (0.75, h.get("high")),
                     (0.90, h.get("wide_high"))], price)
                if not np.isfinite(pct):
                    continue
                key = (tkr, label)
                prev = best.get(key)
                if prev is None or frac > prev["elapsed_frac"]:
                    best[key] = {
                        "horizon": label,
                        "published": payload.get("asof"),
                        "elapsed_frac": round(float(frac), 2),
                        "percentile": round(float(pct), 1),
                        "low": h.get("low"), "high": h.get("high"),
                        "at_edge": int(abs(pct - 50) >= EDGE_PERCENTILE - 50),
                    }

    grouped: dict[str, list] = {}
    for (tkr, _), v in best.items():
        grouped.setdefault(tkr, []).append(v)
    for v in grouped.values():
        v.sort(key=lambda r: -abs(r["percentile"] - 50))
    return grouped


# ----------------------------------------------------------------- direction
# How far the tilt must be from the crowd before it is worth a symbol at all.
LEAN_THRESHOLD_PCT = 0.35


def _tilt(h: dict, close: float) -> float | None:
    """The median forecast against today's price, in log terms.

    Log, not raw, because the price band comes from exponentiating a symmetric
    return band - so its arithmetic midpoint sits above today's price for every
    stock, and reading that as bullish would put an up arrow on all five
    hundred. In log space a flat forecast is flat.
    """
    mid = h.get("mid")
    if not mid or not close or mid <= 0 or close <= 0:
        return None
    return 100.0 * math.log(mid / close)


def _skew(h: dict) -> float | None:
    """Whether the long tail runs up or down, on a -1 to +1 scale.

    Separate from the tilt and worth having: a stock can have a flat median
    with far more room below it than above, which is a different situation from
    one whose whole distribution has shifted.
    """
    lo, mid, hi = h.get("wide_low"), h.get("mid"), h.get("wide_high")
    if not all(v and v > 0 for v in (lo, mid, hi)):
        return None
    up, down = math.log(hi / mid), math.log(mid / lo)
    total = up + down
    if total <= 0:
        return None
    return round(float((up - down) / total), 3)


def direction(records: list[dict]) -> None:
    """Attach a lean per horizon, measured against the day's cross-section.

    Every stock drifts up over a quarter, because the market does. That part is
    real and entirely useless as a signal - it is the same for everything on
    the page and nobody can act on it. So the tilt shown is each stock's median
    forecast MINUS the median tilt across all stocks that day, which leaves
    only what is specific to this name.

    Whether even that carries information is a separate question, and not one
    the arrows can answer. `model.evaluate` measures it directly - hit rate
    against base rate, with a t - and the page prints the result beside the
    arrows so the reader knows what they are worth. If it comes back at zero,
    these should come off the page.
    """
    horizons: dict[str, list] = {}
    for rec in records:
        for label, h in (rec.get("horizons") or {}).items():
            t = _tilt(h, rec.get("close"))
            if t is not None:
                horizons.setdefault(label, []).append(t)

    typical = {k: float(np.median(v)) for k, v in horizons.items() if v}

    for rec in records:
        for label, h in (rec.get("horizons") or {}).items():
            t = _tilt(h, rec.get("close"))
            if t is None:
                continue
            rel = t - typical.get(label, 0.0)
            h["tilt_pct"] = round(float(t), 2)
            h["rel_tilt_pct"] = round(float(rel), 2)
            h["skew"] = _skew(h)
            h["lean"] = ("up" if rel >= LEAN_THRESHOLD_PCT else
                         "down" if rel <= -LEAN_THRESHOLD_PCT else "flat")


# ------------------------------------------------------------------ combine
def attach(records: list[dict], asof: str,
           archive: list[dict] | None = None) -> list[dict]:
    """Add both signals to each record, plus one number to sort the page by."""
    positions = open_ranges(records, asof, archive)
    direction(records)

    for rec in records:
        ts = term_structure(rec)
        if ts:
            rec["term_structure"] = ts
        open_ = positions.get(rec["ticker"]) or []
        if open_:
            rec["open_ranges"] = open_

        # One number for ranking. Position inside a live range is the stronger
        # evidence - something has actually happened - so it leads, and the
        # term-structure stress only breaks ties among names with no archive
        # behind them yet. Scaled so the two are on a comparable footing.
        edge = max((abs(o["percentile"] - 50) for o in open_), default=0.0)
        stress = (ts or {}).get("stress", 0.0)
        rec["watch_score"] = round(float(edge + min(stress, 1.0) * 10.0), 2)
        rec["watch_reason"] = (
            "at the edge of a live range" if edge >= EDGE_PERCENTILE - 50
            else (ts or {}).get("direction", "") if stress >= 0.05
            else "")

    records.sort(key=lambda r: (-r.get("watch_score", 0.0), r["ticker"]))
    return records
