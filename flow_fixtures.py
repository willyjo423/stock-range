"""A synthetic option chain with known answers planted in it.

Every decoy here exists because it is a way the scan could look like it works
while seeing nothing. A test that only proves the flagger flags the obvious
case proves very little; these are built so that a filter quietly doing nothing
shows up as a failure.

* `BLOCK`   - the real thing. At the money, seven days out, $270k of premium
              into a contract with 40 open interest, and all of it arriving
              inside one interval. Must flag.
* `DRIBBLE` - the same premium, spread evenly across the session, into a
              contract with 50,000 open interest. Same dollar figure, opposite
              meaning. Must not flag.
* `FARDATE` - identical to BLOCK but thirty days out.
* `OTM`     - identical to BLOCK but 25% out of the money.
* `WIDE`    - identical to BLOCK but quoted 0.02 bid / 2.00 ask, so its "mid"
              is fiction and the premium built on it is too.
* `SMALL`   - identical to BLOCK at a tenth of the size.
* `QUIET`   - a hundred ordinary contracts that never trade, which is what an
              option chain almost entirely consists of.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-03-02")
STAMP = datetime(2026, 3, 2, 15, 30, tzinfo=timezone.utc)


def occ(root: str, expiry: pd.Timestamp, right: str, strike: float) -> str:
    return (f"{root}{pd.Timestamp(expiry):%y%m%d}{right}"
            f"{int(round(strike * 1000)):08d}")


def _row(ticker, spot, expiry, right, strike, bid, ask, last, volume, oi,
         stamp=None, iv=0.35):
    return {
        "ticker": ticker,
        "contract": occ(ticker, expiry, right, strike),
        "expiry": pd.Timestamp(expiry),
        "right": right,
        "strike": float(strike),
        "bid": float(bid), "ask": float(ask), "last": float(last),
        "volume": float(volume), "open_interest": float(oi),
        "iv": iv, "spot": float(spot),
        "snapshot_at": pd.Timestamp(stamp or STAMP),
        "source": "fixture",
    }


def chain(asof: pd.Timestamp = ASOF, volume_scale: float = 1.0,
          seed: int = 11) -> pd.DataFrame:
    """One snapshot. `volume_scale` shrinks the traded volume for a mid-session
    snapshot, so differencing two of them produces a known burst."""
    asof = pd.Timestamp(asof)
    near = asof + pd.Timedelta(days=7)
    far = asof + pd.Timedelta(days=30)
    rows = []

    # premium = volume x mid x 100 -> 900 x 3.00 x 100 = $270,000
    rows.append(_row("BLOCK", 100.0, near, "C", 100.0, 2.95, 3.05, 3.04,
                     900 * volume_scale, 40))
    # Same premium, but into a crowded contract. vol_oi = 0.018.
    rows.append(_row("DRIBBLE", 100.0, near, "C", 100.0, 2.95, 3.05, 3.00,
                     900 * (0.5 + 0.5 * volume_scale), 50_000))
    rows.append(_row("FARDATE", 100.0, far, "C", 100.0, 2.95, 3.05, 3.04,
                     900 * volume_scale, 40))
    rows.append(_row("OTM", 100.0, near, "C", 125.0, 2.95, 3.05, 3.04,
                     900 * volume_scale, 40))
    rows.append(_row("WIDE", 100.0, near, "C", 100.0, 0.02, 2.00, 1.90,
                     900 * volume_scale, 40))
    rows.append(_row("SMALL", 100.0, near, "C", 100.0, 2.95, 3.05, 3.04,
                     90 * volume_scale, 40))

    # A straddle: matched call and put volume, which is not a directional bet.
    rows.append(_row("STRADDLE", 50.0, near, "C", 50.0, 1.95, 2.05, 2.00,
                     600 * volume_scale, 30))
    rows.append(_row("STRADDLE", 50.0, near, "P", 50.0, 1.95, 2.05, 2.00,
                     600 * volume_scale, 30))

    # A put-side flag, so the roll-up has something one-sided to tilt on.
    rows.append(_row("PUTSIDE", 80.0, near, "P", 80.0, 2.45, 2.55, 2.46,
                     800 * volume_scale, 25))

    rng = np.random.default_rng(seed)
    for i in range(100):
        strike = 100.0 + (i % 21 - 10) * 2.5
        exp = near if i % 2 else far
        right = "C" if i % 3 else "P"
        rows.append(_row("QUIET", 100.0, exp, right, strike,
                         1.20, 1.30, 1.25, float(rng.integers(0, 6)),
                         float(rng.integers(50, 4000))))
    return pd.DataFrame(rows)


def two_snapshots(asof: pd.Timestamp = ASOF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Morning and afternoon.

    BLOCK trades nothing in the morning and all of it in the afternoon, so its
    burst is 1.0. DRIBBLE trades half and half, so its burst is 0.5. That is
    the only free way to tell them apart and this is the test of it.
    """
    return chain(asof, volume_scale=0.0), chain(asof, volume_scale=1.0)


# ------------------------------------------------------- grading fixtures
def graded_frame(n_days: int = 40, n_names: int = 60, flag_rate: float = 0.06,
                 effect: float = 0.0, seed: int = 7) -> pd.DataFrame:
    """What `flow_grade.build()` produces, with a controllable planted effect.

    `effect` is how much wider the flagged names' moves are, in band
    half-widths. At 0.0 the flags carry no information at all and the grader
    must say so; the tests check both directions, because a grader that always
    finds something is worse than no grader.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-01-05", periods=n_days, freq="B")
    rows = []
    for d in days:
        for i in range(n_names):
            flagged = int(rng.random() < flag_rate)
            # A calibrated band means the move is standard-normal-ish in band
            # half-widths, scaled so that |z| > 1 happens about half the time.
            z = rng.normal(0.0, 1.5) * (1.0 + effect * flagged)
            tilt = ("calls" if rng.random() < 0.5 else "puts") if flagged else None
            rows.append({
                "asof": d,
                "settle_date": d + pd.Timedelta(days=7),
                "ticker": f"T{i:03d}",
                "flagged": flagged,
                "flag_score": float(rng.uniform(1, 6)) if flagged else np.nan,
                "tilt": tilt,
                "multileg": 0,
                "oi_confirmed": float(rng.random()) if flagged else np.nan,
                "move_z": z,
                "abs_move_z": abs(z),
                "breakout": int(abs(z) > 1.0),
                "above": int(z > 1.0),
                "below": int(z < -1.0),
                "inside": bool(abs(z) <= 1.0),
            })
    return pd.DataFrame(rows)
