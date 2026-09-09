"""Feature construction.

Everything here is computable from data up to and including the row's own
date, and nothing looks past it. The single exception is the target, which is
built in `prices.forward_return` and named plainly there.

The design follows from what the probe measured. The naive band - assume the
next month looks like the last one - lands at 50% overall but falls apart
underneath: at a one-week horizon it covered 31% of outcomes for the calmest
quarter of stocks and 64% for the wildest. That is volatility reverting to its
mean. A quiet stock does not stay as quiet as its trailing window suggests, and
a wild one calms down.

So the features deliberately give the model both ends of that: short windows
(what is happening now) and long ones (what normal looks like for this name),
so it can blend them rather than trusting either alone. That blend is the whole
job, and it is why `rv_252` matters as much as `rv_5`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
import prices

log = logging.getLogger(__name__)

# --- feature groups, each switchable so each can be measured -----------------

# The HAR core: the same quantity over several horizons. This is the baseline
# every other group has to improve on, and it is what carries the mean
# reversion the probe exposed.
VOL_COLUMNS = [
    "log_rv_5", "log_rv_21", "log_rv_63", "log_rv_252",
    "rv_slope_short", "rv_slope_long",
]

# Volatility is not one number. A stock can be quiet on close-to-close and
# violent intraday, or calm all day and gap at the open, and those behave
# differently going forward.
SHAPE_COLUMNS = [
    "range_vol_21", "gap_vol_21", "downside_vol_21", "upside_vol_21",
    "skew_63", "kurt_63", "max_abs_ret_21",
]

# Where the whole market is. An individual stock's calm means less when the
# index is falling apart.
MARKET_COLUMNS = [
    "log_vix", "vix_chg_21", "log_spy_rv_21", "beta_63", "rel_vol",
]

# Liquidity and price level, both of which are known to track volatility.
FLOW_COLUMNS = [
    "log_dollar_volume_21", "dollar_volume_chg", "log_price",
    "trailing_ret_21", "trailing_ret_63", "trailing_ret_252",
]

# Horizon-specific: whether the window being forecast contains a report. The
# probe found dates going back a median of 12.3 years, deep enough to train on.
EARNINGS_COLUMNS = ["days_to_earnings", "earnings_in_window", "earnings_count"]

FEATURE_GROUPS = {
    "vol": VOL_COLUMNS,
    "shape": SHAPE_COLUMNS,
    "market": MARKET_COLUMNS,
    "flow": FLOW_COLUMNS,
    "earnings": EARNINGS_COLUMNS,
}

FEATURE_COLUMNS = (VOL_COLUMNS + SHAPE_COLUMNS + MARKET_COLUMNS
                   + FLOW_COLUMNS + EARNINGS_COLUMNS)

# Never features, carried for evaluation and display.
CARRY = ["ticker", "date", "close", "scale", "fwd_ret", "z", "rv_ref"]


def _safe_log(s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0))


def build_panel(px: pd.DataFrame, market: pd.DataFrame | None = None
                ) -> pd.DataFrame:
    """Every horizon-independent feature, one row per ticker per day."""
    df = prices.realized_vol(prices.add_returns(px))
    g = df.groupby("ticker")

    for w in config.RV_WINDOWS:
        df[f"log_rv_{w}"] = _safe_log(df[f"rv_{w}"])

    # Two slopes rather than one: the very short against the medium, and the
    # medium against the long. A stock can be spiking this week while still
    # calm by its own standards, and those are different situations.
    df["rv_slope_short"] = df["log_rv_5"] - df["log_rv_21"]
    df["rv_slope_long"] = df["log_rv_21"] - df["log_rv_252"]

    df["range_vol_21"] = (g["range_vol"].rolling(21, min_periods=10).mean()
                           .reset_index(level=0, drop=True) * np.sqrt(252))
    df["gap_vol_21"] = (g["gap"].rolling(21, min_periods=10).std()
                         .reset_index(level=0, drop=True) * np.sqrt(252))

    # Downside and upside separately. Falling markets are more volatile than
    # rising ones, and a single standard deviation cannot say so.
    down = df["ret"].where(df["ret"] < 0)
    up = df["ret"].where(df["ret"] > 0)
    df["downside_vol_21"] = (down.groupby(df["ticker"])
                             .rolling(21, min_periods=5).std()
                             .reset_index(level=0, drop=True) * np.sqrt(252))
    df["upside_vol_21"] = (up.groupby(df["ticker"])
                           .rolling(21, min_periods=5).std()
                           .reset_index(level=0, drop=True) * np.sqrt(252))

    df["skew_63"] = (g["ret"].rolling(63, min_periods=30).skew()
                      .reset_index(level=0, drop=True))
    df["kurt_63"] = (g["ret"].rolling(63, min_periods=30).kurt()
                      .reset_index(level=0, drop=True))
    df["max_abs_ret_21"] = (g["abs_ret"].rolling(21, min_periods=10).max()
                             .reset_index(level=0, drop=True))

    dv = df["close"] * df["volume"]
    df["log_dollar_volume_21"] = _safe_log(
        dv.groupby(df["ticker"]).rolling(21, min_periods=10).mean()
          .reset_index(level=0, drop=True))
    df["dollar_volume_chg"] = (
        df["log_dollar_volume_21"]
        - df.groupby("ticker")["log_dollar_volume_21"].shift(21))
    df["log_price"] = _safe_log(df["close"])

    for w in (21, 63, 252):
        df[f"trailing_ret_{w}"] = (
            _safe_log(df["close"]) - _safe_log(df.groupby("ticker")["close"].shift(w)))

    df = _add_market(df, market)
    return df


def _add_market(df: pd.DataFrame, market: pd.DataFrame | None) -> pd.DataFrame:
    """Index-level context, joined on date.

    Absent market data leaves these NaN rather than zero. The booster handles
    missing natively, and a zero here would assert that the VIX was at 1.0.
    """
    for c in MARKET_COLUMNS:
        df[c] = np.nan
    if market is None or market.empty:
        log.warning("no market reference data; market features will be empty")
        return df

    m = prices.realized_vol(prices.add_returns(market))
    vix = m[m["ticker"] == "^VIX"][["date", "close"]].rename(
        columns={"close": "vix"})
    spy = m[m["ticker"] == "SPY"][["date", "ret", "rv_21"]].rename(
        columns={"ret": "spy_ret", "rv_21": "spy_rv_21"})

    ref = vix.merge(spy, on="date", how="outer").sort_values("date")
    ref["log_vix"] = _safe_log(ref["vix"] / 100.0)
    ref["vix_chg_21"] = ref["log_vix"] - ref["log_vix"].shift(21)
    ref["log_spy_rv_21"] = _safe_log(ref["spy_rv_21"])

    df = df.drop(columns=[c for c in MARKET_COLUMNS if c in df.columns])
    df = df.merge(ref[["date", "log_vix", "vix_chg_21", "log_spy_rv_21",
                       "spy_rv_21", "spy_ret"]], on="date", how="left")

    # Rolling beta to the index: covariance over variance, both trailing.
    df = df.sort_values(["ticker", "date"])
    gb = df.groupby("ticker")
    cov = (gb.apply(lambda d: d["ret"].rolling(63, min_periods=30)
                    .cov(d["spy_ret"]), include_groups=False)
             .reset_index(level=0, drop=True))
    var = (gb["spy_ret"].rolling(63, min_periods=30).var()
             .reset_index(level=0, drop=True))
    df["beta_63"] = (cov / var.replace(0, np.nan)).to_numpy()
    df["rel_vol"] = df["rv_21"] / df["spy_rv_21"].replace(0, np.nan)
    return df.drop(columns=["spy_ret", "spy_rv_21"])


def add_earnings(df: pd.DataFrame, earnings: pd.DataFrame | None,
                 horizon: int) -> pd.DataFrame:
    """Whether a report lands inside the window being forecast.

    This is the one feature that must be rebuilt per horizon: a report eight
    days out is inside a one-month window and outside a one-week one, and that
    is precisely the distinction worth drawing.
    """
    out = df.copy()
    for c in EARNINGS_COLUMNS:
        out[c] = np.nan
    if earnings is None or earnings.empty:
        return out

    e = earnings.dropna(subset=["ticker", "date"]).copy()
    e["date"] = pd.to_datetime(e["date"]).dt.tz_localize(None)
    e = e.sort_values(["ticker", "date"])

    # Trading days are roughly 21 per month; the window is in trading days but
    # earnings dates are calendar dates, so convert with that ratio.
    span = pd.Timedelta(days=int(round(horizon * 365.25 / 252)))

    pieces = []
    for tkr, block in out.groupby("ticker", sort=False):
        dates = e.loc[e["ticker"] == tkr, "date"].to_numpy()
        block = block.sort_values("date").copy()
        if len(dates) == 0:
            pieces.append(block)
            continue
        own = block["date"].to_numpy()
        nxt = np.searchsorted(dates, own, side="left")
        has_next = nxt < len(dates)
        days = np.full(len(own), np.nan)
        days[has_next] = (dates[nxt[has_next]] - own[has_next]) / np.timedelta64(1, "D")
        block["days_to_earnings"] = days
        end = own + span.to_timedelta64()
        upto = np.searchsorted(dates, end, side="right")
        count = upto - nxt
        block["earnings_count"] = count.astype(float)
        block["earnings_in_window"] = (count > 0).astype(float)
        pieces.append(block)

    return pd.concat(pieces, ignore_index=True)


def build_for_horizon(panel: pd.DataFrame, horizon: int,
                      earnings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the target and the scale the model works in.

    The target is not the forward return itself but the forward return divided
    by the naive band width - what the move would be if the next `horizon` days
    behaved exactly like the trailing window. Predicting quantiles of *that*
    means the trees only have to learn how the shape differs from the naive
    assumption, not relearn the level of volatility from scratch. It is the
    same trick as riding the football model on a rescaled ratings baseline, and
    it is what lets a quantile model extrapolate to a stock wilder than
    anything in its training data.
    """
    df = add_earnings(panel, earnings, horizon)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ref = f"rv_{horizon}" if f"rv_{horizon}" in df.columns else "rv_21"
    df["rv_ref"] = df[ref]
    df["scale"] = df["rv_ref"] * np.sqrt(horizon / 252.0)
    # A zero scale would divide the target to infinity; a stock that genuinely
    # did not move at all carries no information about how far it might.
    df.loc[df["scale"] <= 1e-6, "scale"] = np.nan

    df["fwd_ret"] = prices.forward_return(df, horizon)
    df["z"] = df["fwd_ret"] / df["scale"]

    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def training_matrix(df: pd.DataFrame, columns: list[str] | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = columns or FEATURE_COLUMNS
    usable = df["z"].notna() & df["scale"].notna() & np.isfinite(df["z"])
    # A handful of rows carry absurd standardized moves, almost always a stock
    # that was dead quiet and then had one enormous day. They are real, but a
    # few of them can dominate a quantile fit, so the tails are trimmed at a
    # level far outside anything the model is asked to predict.
    usable &= df["z"].abs() < 25
    keep = df.loc[usable].copy()
    carry = [c for c in CARRY if c in keep.columns]
    return keep[cols], keep[carry]
