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
# The first probe run 404'd on both of these - the file gets renamed with each
# refresh, so a pinned name rots. They are kept as a fast path and the real
# answer now comes from reconstructing membership backwards through the
# "Selected changes" table on the Wikipedia page, which is fetched anyway.
SP500_HISTORY_URLS = [
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes(current).csv",
    "https://github.com/fja05680/sp500/raw/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes(current).csv",
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv",
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
