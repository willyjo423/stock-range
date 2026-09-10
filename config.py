"""Central configuration for the range-forecasting build.

The question this project answers is narrow on purpose: *how wide* is a stock's
plausible range over the next week, month or quarter. Not which direction.

That is not modesty, it is where the signal is. Volatility clusters strongly -
a stock that has been swinging keeps swinging - so dispersion is forecastable.
Drift over a few weeks is a fraction of a percent and drowns in noise, so
direction is not. The sports models spent their effort on a point forecast and
wrapped a band around it; here the band is the whole product.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
FORECASTS = DATA / "forecasts"

for _p in (DATA, CACHE, MODELS, DOCS, FORECASTS):
    _p.mkdir(parents=True, exist_ok=True)

# --- Universe --------------------------------------------------------------
# Current membership, and a history of changes so a backtest is not run only on
# the companies that survived to today. See universe.py for why that matters.
SP500_CURRENT_URLS = [
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
]
# Full membership on every change date, as `date,tickers`. The name matters:
# the first live run pinned "(current)" and got a 404 on both spellings, then
# fell through to the plain file - which is a frozen 1996-2019 snapshot, so
# membership silently stopped moving in January 2019 for seven years. The
# maintained file is "(Updated)", with a space before the bracket, and it goes
# first. The plain one stays as a floor, because a stale record beats none.
SP500_HISTORY_URLS = [
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv",
    "https://github.com/fja05680/sp500/raw/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv",
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv",
]

# And the additions and removals since the plain file ends, as
# `date,add,remove` with comma-separated tickers inside each cell. This is the
# belt to the braces above: if the "(Updated)" file is ever stale, these
# changes are applied forward from wherever the published record stops, so the
# gap closes either way.
#
# Applying changes forward from a known state is also sounder than the
# alternative that was in here - walking today's index backwards through a
# scraped Wikipedia table. Forward from a record cannot be wrong about where it
# started; backward from today is only as good as today's scrape.
SP500_CHANGES_URLS = [
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "sp500_changes_since_2019.csv",
    "https://github.com/fja05680/sp500/raw/master/"
    "sp500_changes_since_2019.csv",
]

# A reconstruction is only trusted if the membership count stays near 500 at
# every point in history. Drift outside this band means changes are being
# missed or double-applied, and the probe says so rather than quietly training
# on a 380-name index.
MEMBERSHIP_COUNT_BAND = (460, 530)

# Index and volatility references, used as context features and as the free
# benchmark. No account needed for any of them.
BENCHMARK_TICKERS = ["SPY", "QQQ", "IWM", "^VIX", "^VXN", "^GSPC"]

# --- History ---------------------------------------------------------------
# 2004 is a deliberate floor. It keeps 2008 and 2020 in, which is essential -
# a volatility model trained only on calm years will be confidently wrong in
# the first storm - while staying inside the era of decimal pricing and modern
# market structure.
HISTORY_START = os.environ.get("HISTORY_START", "2004-01-01")

# --- Horizons --------------------------------------------------------------
# Trading days, not calendar days: 5 is a week, 21 a month, 63 a quarter.
# Each horizon gets its own model. Interpolating one model across horizons
# sounds tidy and quietly breaks calibration at the ends.
HORIZONS = {"1 week": 5, "1 month": 21, "3 months": 63}

# The quantiles the model predicts directly, rather than deriving from a
# standard deviation. Stock returns are fat-tailed and left-skewed, so a
# symmetric band around a mean misdescribes both ends - and understates the
# downside, which is the end that matters.
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
BAND = (0.25, 0.75)   # the middle half, the headline range

# --- Volatility features ---------------------------------------------------
# Trailing realized volatility windows, in trading days. This trio is the HAR
# structure - daily, weekly, monthly - which is a stubbornly hard baseline to
# beat and is the closest thing here to the power ratings in the sports builds.
RV_WINDOWS = [5, 21, 63, 252]

# A stock needs this much history before it can be forecast at all.
MIN_HISTORY_DAYS = 300

# A daily move larger than this is almost certainly an unadjusted corporate
# action rather than a real return. The probe counts them; the loader does not
# silently discard them, because a genuine 50% gap is real information and
# deciding which is which is not the loader's job.
IMPLAUSIBLE_DAILY_MOVE = 0.40

# --- Runtime ---------------------------------------------------------------
RANDOM_SEED = 1729
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
DOWNLOAD_CHUNK = 40          # tickers per request; 500 at once gets throttled
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL", 12 * 3600))

# --- Options flow ----------------------------------------------------------
# A separate product from the ranges, sharing this project's universe and -
# more importantly - its bands, which are what make the flow scan gradeable.
#
# The honest framing first, because every filter below is shaped by it. Real
# options flow is a TAPE: every print, with its size, its price, and the quote
# that stood at the instant it happened. That is OPRA data and it is licensed.
# Nothing free carries it. What free sources carry is a SNAPSHOT of the chain -
# strike, bid, ask, last, cumulative day volume, open interest, IV - refreshed
# on a delay.
#
# So a single $80k sweep lifting the offer and four hundred retail lots
# trickling out at the mid arrive here as the same number. The one free lever
# against that is time: snapshot often enough and a block shows up as almost
# all of the day's volume appearing inside one interval, while the dribble
# spreads evenly. That is what FLOW_BURST measures, and it is the reason this
# runs several times a day rather than once after the close.

FLOW = DATA / "flow"
FLOW_SNAPSHOTS = FLOW / "snapshots"     # raw chains, one file per scan
FLOW_FLAGS = FLOW / "flags"             # what was flagged, for forward grading

for _p in (FLOW, FLOW_SNAPSHOTS, FLOW_FLAGS):
    _p.mkdir(parents=True, exist_ok=True)

# CBOE publishes a full delayed chain per underlying as one JSON document -
# every expiry in a single request, where yfinance needs one request per
# expiry. For 500 names that is the difference between a scan that finishes
# inside a schedule slot and one that does not. yfinance stays as the fallback.
# Neither has been confirmed against the live endpoint yet; flow_probe.py is
# what confirms them, and it should be run before any of this is believed.
CBOE_CHAIN_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
CBOE_INDEX_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_{sym}.json"

# --- The four filters ------------------------------------------------------
# Expirations inside two weeks. Short-dated is where a directional bet has to
# be right about timing as well as direction, so it is where conviction shows.
FLOW_MAX_DTE = 14
# And a floor, because a contract expiring today trades on gamma mechanics
# rather than on anyone's view, and 0-DTE volume would swamp everything else.
FLOW_MIN_DTE = 1

# "On the money". Measured in log terms so a 5% band means the same thing
# above and below the strike.
FLOW_ATM_BAND = 0.05

# Premium floor, in dollars. Note what this is: volume x mid x 100 for the
# interval, which is TOTAL premium traded in that contract, not the size of
# any one trade. It is the weakest of the four filters and the page says so.
FLOW_MIN_PREMIUM = 50_000.0

# Low open interest, two ways, and both are needed.
#
# Absolute, which is the criterion as stated: few contracts outstanding going
# in, so what trades today is not people shuffling an existing position.
FLOW_MAX_OI = 1_000
# And relative, which is the stronger form: today's volume against the open
# interest that stood before it. Above 1.0 means more contracts changed hands
# today than existed yesterday, which is hard to explain as anything but new
# positioning. Worth knowing: the open interest in an intraday snapshot is
# LAST NIGHT'S settled figure - it does not tick during the session - so this
# ratio is correctly computed against a number that predates today's trading.
FLOW_MIN_VOL_OI = 1.0

# --- Quality gates ---------------------------------------------------------
# A contract quoted 0.05 bid / 0.90 ask has no meaningful mid, and premium
# estimated from that mid is fiction.
FLOW_MAX_SPREAD_PCT = 0.35
# A quote of zero bid is a contract nobody wants; its "last" is stale by
# construction.
FLOW_MIN_BID = 0.05

# Share of a contract's day volume that arrived inside a single scan interval.
# High means concentrated - the free proxy for a block. It needs at least two
# snapshots in a session to mean anything, and is reported as unknown until it
# has them.
FLOW_BURST_ALERT = 0.60

# How many scans a day. Three is the default: mid-morning, midday, and just
# after the close for the settled picture. More is better for burst detection
# and worse for rate limits.
FLOW_SCANS_PER_DAY = 3

# Requests are spaced to stay under the free endpoints' patience.
FLOW_REQUEST_PAUSE = 0.25
FLOW_MAX_WORKERS = 8

# --- Grading ---------------------------------------------------------------
# The horizon a flag is graded over. Two weeks of expiry means the thesis, if
# there is one, should show inside a week.
FLOW_GRADE_HORIZON = "1 week"
# Flags below this score are recorded but not shown on the page. They are
# still graded, which is the point - a threshold nobody tested is a guess.
FLOW_SHOW_MIN_SCORE = 1.0
