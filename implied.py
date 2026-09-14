"""What the option market is charging, on the same footing as the model's range.

The whole point of this file is one comparison, and it only means anything if
both sides are measured the same way.

**The model's range is the middle half.** "Half the time between $324 and $340"
is the 25th to 75th percentile. In log terms its half-width is 0.6745 standard
deviations, because that is where the quartiles of a normal sit. Dividing by
0.6745 turns it into a one-standard-deviation move.

**An option's implied move is already one standard deviation**, near enough -
implied volatility is quoted annualised, so scaling by the square root of the
horizon gives the move over that horizon.

Comparing the two without that conversion is the obvious way to get this wrong,
and it fails in a specific direction: the middle-half band is about a third
narrower than a one-sigma move, so every stock on the page would look like its
options were expensive. A page where the answer is always "expensive" is not a
signal, it is a unit error.

There is a second unit trap underneath. Implied volatility is quoted per
*calendar* year but the model's horizons are in *trading* days, and 21 trading
days is a month of calendar time but only 21/252 of a trading year. Both sides
here are converted to trading-day terms, so a week means the same thing twice.

What this is not
----------------
It is not a claim that the model is right and the market is wrong. The market
prices things the model cannot see - a pending lawsuit, a rumoured deal, a
product launch. A gap is a question worth asking, not an answer. Whether these
gaps predict anything is measurable, and `archive()` writes every one of them
down so that it can be checked later rather than believed now.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

# Where the quartiles of a normal distribution sit, in standard deviations.
# The middle half of outcomes spans plus or minus this much.
QUARTILE_Z = 0.6744897501960817

# An at-the-money straddle costs about this many standard deviations of the
# underlying, for a lognormal. Used only as a cross-check on the quoted implied
# volatility - if the two disagree badly, the quotes are unreliable.
STRADDLE_TO_SIGMA = math.sqrt(2.0 / math.pi)      # 0.7979

TRADING_DAYS = 252

# How far from the money a contract may sit and still be treated as at-the-money
# for the purpose of reading an implied volatility off it.
ATM_TOLERANCE = 0.06
# Below this the quote is not worth reading.
MIN_BID = 0.02
MAX_SPREAD_PCT = 0.60
# Implied volatility outside this band is a data error, not a market view.
IV_BOUNDS = (0.03, 5.0)

# How far the two sides have to diverge before the page says anything about it.
# These are presentation thresholds, not measured edges - see the note in
# `verdict` below.
RICH_RATIO = 1.15
CHEAP_RATIO = 0.87


# ------------------------------------------------------------ the model side
def model_sigma_pct(low: float, high: float) -> float | None:
    """Turn a published middle-half range into a one-sigma move, in percent.

    Log terms, because the band came from exponentiating a return distribution.
    Reading it linearly would overstate the width of every high-priced stock.
    """
    if not low or not high or low <= 0 or high <= 0 or high <= low:
        return None
    half_width = (math.log(high) - math.log(low)) / 2.0
    return 100.0 * half_width / QUARTILE_Z


# ---------------------------------------------------------- the market side
def _atm_iv(chain: pd.DataFrame, spot: float) -> tuple[float | None, float | None]:
    """Implied volatility at the money, and the straddle cross-check.

    Takes the call and the put closest to the spot price, requires both to have
    a quote worth reading, and averages their implied volatilities. Averaging
    the two sides matters: a single side can be skewed by demand for protection,
    and the question here is how much movement is priced, not which way.
    """
    if chain is None or chain.empty or not spot or spot <= 0:
        return None, None

    usable = chain[
        (chain["bid"] >= MIN_BID)
        & (chain["ask"] > chain["bid"])
        & (chain["spread_pct"] <= MAX_SPREAD_PCT)
        & chain["iv"].between(*IV_BOUNDS)
        & (chain["log_moneyness"].abs() <= math.log(1 + ATM_TOLERANCE))
    ]
    if usable.empty:
        return None, None

    ivs, straddle = [], 0.0
    for right in ("C", "P"):
        side = usable[usable["right"] == right]
        if side.empty:
            return None, None
        nearest = side.iloc[side["log_moneyness"].abs().to_numpy().argmin()]
        ivs.append(float(nearest["iv"]))
        straddle += float(nearest["mid"])

    iv = float(np.mean(ivs))
    return iv, (straddle / spot if straddle > 0 else None)


def implied_sigma_pct(chain: pd.DataFrame, spot: float, horizon_days: int,
                      dte: int | None = None) -> dict:
    """A one-sigma move over `horizon_days` trading days, from the chain.

    The expiry used is whichever sits closest to the horizon, so the term
    structure is respected - a one-week number should come from a one-week
    option, not from a quarterly one. But the volatility read off it is then
    applied at the model's exact horizon, so the two sides describe the same
    stretch of time even when no expiry lines up neatly.
    """
    iv, straddle_move = _atm_iv(chain, spot)
    if iv is None:
        return {}

    years = horizon_days / TRADING_DAYS
    sigma = 100.0 * iv * math.sqrt(years)

    out = {"implied_move_pct": round(sigma, 2),
           "iv": round(iv, 4),
           "expiry_dte": int(dte) if dte is not None else None}

    # The straddle is priced over the life of the contract it belongs to, so it
    # only cross-checks the volatility when that contract is near the horizon.
    if straddle_move and dte:
        straddle_sigma = straddle_move / STRADDLE_TO_SIGMA
        implied_at_expiry = iv * math.sqrt(max(dte, 1) / 365.0)
        if implied_at_expiry > 0:
            agree = straddle_sigma / implied_at_expiry
            out["straddle_check"] = round(float(agree), 2)
            # Well outside 1.0 means the quotes and the quoted vol disagree,
            # which is a data problem rather than a market view.
            out["quotes_ok"] = bool(0.7 <= agree <= 1.4)
    return out


def pick_expiry(chain: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, int | None]:
    """The listed expiry nearest the horizon, in calendar terms."""
    if chain is None or chain.empty:
        return chain, None
    target_cal = horizon_days * 365.0 / TRADING_DAYS
    dtes = chain["dte"].dropna().unique()
    future = [d for d in dtes if d >= 1]
    if not future:
        return chain.iloc[0:0], None
    best = min(future, key=lambda d: abs(d - target_cal))
    return chain[chain["dte"] == best], int(best)


# ------------------------------------------------------------------ compare
def verdict(ratio: float | None) -> str:
    """A label for the gap.

    The thresholds are chosen for readability, not measured. Whether a stock
    whose options are priced 20% above the model's expectation actually moves
    less than priced is an open question, and `archive` writes every comparison
    down so it can be answered from the forward record instead of assumed.
    """
    if ratio is None or not np.isfinite(ratio):
        return "unknown"
    if ratio >= RICH_RATIO:
        return "expensive"
    if ratio <= CHEAP_RATIO:
        return "cheap"
    return "fair"


def compare(model_pct: float | None, implied: dict) -> dict:
    if not model_pct or not implied.get("implied_move_pct"):
        return {}
    imp = implied["implied_move_pct"]
    ratio = imp / model_pct
    return {
        "model_move_pct": round(model_pct, 2),
        "implied_move_pct": round(imp, 2),
        "ratio": round(float(ratio), 3),
        "gap_pct": round(100.0 * (ratio - 1.0), 0),
        "verdict": verdict(ratio),
        "iv": implied.get("iv"),
        "expiry_dte": implied.get("expiry_dte"),
        "quotes_ok": implied.get("quotes_ok", True),
    }


# --------------------------------------------------------------------- lean
# How far the cross-sectional tilt has to be from the crowd to count as a vote.
TILT_VOTE_PCT = 0.30
# And how lopsided the flow has to be.
FLOW_VOTE_SHARE = 0.65
# A tail this much longer on one side counts as a vote.
SKEW_VOTE = 0.08


def lean(horizon: dict, flow: dict | None) -> dict:
    """One direction, and the three things that voted for it.

    The inputs are deliberately not averaged into a score. They carry very
    different amounts of evidence - the model tilt has a measured hit rate
    behind it, the flow has none yet - and a weighted average would bury that
    difference under a single number. So each votes, and when they disagree the
    answer is that they disagree.
    """
    votes, drivers = [], []

    tilt = horizon.get("rel_tilt_pct")
    if tilt is not None:
        v = 1 if tilt >= TILT_VOTE_PCT else -1 if tilt <= -TILT_VOTE_PCT else 0
        votes.append(v)
        drivers.append({
            "name": "model tilt",
            "direction": "up" if v > 0 else "down" if v < 0 else "flat",
            "detail": f"{tilt:+.1f}% against the day's typical stock",
            "evidence": "measured",
        })

    if flow:
        share = flow.get("call_share")
        if share is not None:
            if flow.get("looks_multileg"):
                v = 0
                detail = ("matched call and put legs - reads as a straddle, "
                          "not a direction")
            else:
                v = (1 if share >= FLOW_VOTE_SHARE
                     else -1 if share <= 1 - FLOW_VOTE_SHARE else 0)
                detail = (f"${(flow.get('call_premium') or 0) / 1e6:.1f}m calls "
                          f"vs ${(flow.get('put_premium') or 0) / 1e6:.1f}m puts")
            votes.append(v)
            drivers.append({
                "name": "options flow",
                "direction": "up" if v > 0 else "down" if v < 0 else "flat",
                "detail": detail,
                "evidence": "ungraded",
            })

    sk = horizon.get("skew")
    if sk is not None:
        v = 1 if sk >= SKEW_VOTE else -1 if sk <= -SKEW_VOTE else 0
        drivers.append({
            "name": "shape",
            "direction": "up" if v > 0 else "down" if v < 0 else "flat",
            "detail": ("more room above than below" if v > 0 else
                       "more room below than above" if v < 0 else
                       "symmetric - no tail preference"),
            "evidence": "ungraded",
        })
        votes.append(v)

    live = [v for v in votes if v != 0]
    if not live:
        direction, strength = "flat", "nothing to say"
    elif len(set(np.sign(live))) > 1:
        direction, strength = "flat", "signals disagree"
    else:
        direction = "up" if live[0] > 0 else "down"
        strength = "weak" if len(live) == 1 else "moderate"

    return {"direction": direction, "strength": strength, "drivers": drivers}


# ------------------------------------------------------------------ archive
def archive(payload: dict) -> str | None:
    """Write today's comparisons down so they can be graded later.

    Nothing here has been shown to predict anything. The only way that changes
    is if every comparison is recorded before the outcome is known, which is
    what this does. A gap believed without a record is a hunch with a chart
    attached.
    """
    if not payload.get("tickers"):
        return None
    folder = config.DATA / "implied"
    folder.mkdir(parents=True, exist_ok=True)
    slim = {
        "asof": payload.get("asof"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers": [
            {"ticker": r["ticker"], "close": r.get("close"),
             "horizons": {k: {kk: v.get(kk) for kk in
                              ("model_move_pct", "implied_move_pct", "ratio",
                               "verdict", "iv", "expiry_dte")}
                          for k, v in (r.get("horizons") or {}).items()
                          if v.get("ratio") is not None},
             "lean": (r.get("lean") or {}).get("direction")}
            for r in payload["tickers"]],
    }
    path = folder / f"{payload['asof']}.json"
    path.write_text(json.dumps(slim, indent=2))
    return str(path)
