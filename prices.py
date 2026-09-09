"""Daily prices, and the realized volatility built from them.

Two sources, tried in order, because one of them is always having a bad day:

* **yfinance** - decades of split- and dividend-adjusted daily bars, batched.
* **Stooq** - a plain CSV per ticker, no key, no library. Slower for hundreds
  of names but it answers when Yahoo does not.

The one thing that matters more than which source answers is **adjustment**. A
2-for-1 split in unadjusted data looks like a 50% crash, and a volatility model
fed those learns that stocks are twice as wild as they are. Everything here
uses adjusted closes, and `audit()` counts the moves too large to be real so a
silent adjustment failure shows up as a number rather than as a slightly worse
model six weeks later.
"""
from __future__ import annotations

import hashlib
import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config

log = logging.getLogger(__name__)


class PricesUnavailable(RuntimeError):
    """No source returned usable bars."""


# ---------------------------------------------------------------- caching
def _cache_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return config.CACHE / f"px_{digest}.pkl.gz"


def _cached(key: str, ttl: int | None):
    path = _cache_path(key)
    if not path.exists():
        return None
    if ttl is not None and (time.time() - path.stat().st_mtime) > ttl:
        return None
    try:
        return pd.read_pickle(path, compression="gzip")
    except Exception as exc:  # noqa: BLE001 - a bad cache entry is not fatal
        log.warning("unreadable cache %s: %s", path.name, exc)
        return None


def _store(key: str, df: pd.DataFrame) -> None:
    try:
        df.to_pickle(_cache_path(key), compression="gzip")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not cache %s: %s", key, exc)


# ---------------------------------------------------------------- sources
def _from_yfinance(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), config.DOWNLOAD_CHUNK):
        chunk = tickers[i:i + config.DOWNLOAD_CHUNK]
        raw = yf.download(chunk, start=start, auto_adjust=True,
                          progress=False, group_by="ticker", threads=True)
        if raw is None or raw.empty:
            log.warning("yfinance returned nothing for chunk %d", i)
            continue

        # One ticker comes back flat, several come back with a MultiIndex.
        if isinstance(raw.columns, pd.MultiIndex):
            for t in chunk:
                if t not in raw.columns.get_level_values(0):
                    continue
                sub = raw[t].dropna(how="all")
                if sub.empty:
                    continue
                frames.append(_tidy(sub, t))
        else:
            frames.append(_tidy(raw.dropna(how="all"), chunk[0]))
        log.info("yfinance: %d/%d tickers", min(i + len(chunk), len(tickers)),
                 len(tickers))

    if not frames:
        raise PricesUnavailable("yfinance returned nothing")
    return pd.concat(frames, ignore_index=True)


def _from_stooq(tickers: list[str], start: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        sym = t.lower().replace("-", ".").lstrip("^")
        url = f"https://stooq.com/q/d/l/?s={sym}.us&i=d"
        try:
            resp = requests.get(url, timeout=config.REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            log.warning("stooq %s: %s", t, exc)
            continue
        if resp.status_code != 200 or len(resp.content) < 100:
            continue
        try:
            df = pd.read_csv(io.BytesIO(resp.content))
        except Exception:  # noqa: BLE001
            continue
        if "Close" not in df.columns:
            continue
        frames.append(_tidy(df.set_index("Date"), t))
    if not frames:
        raise PricesUnavailable("stooq returned nothing")
    return pd.concat(frames, ignore_index=True)


def _tidy(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.reset_index()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "date"})
    keep = {c: c for c in ("date", "open", "high", "low", "close", "volume")
            if c in out.columns}
    out = out[list(keep)].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    out["ticker"] = ticker
    for c in ("open", "high", "low", "close", "volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan
    return out.dropna(subset=["date", "close"])


def load(tickers: list[str], start: str | None = None,
         ttl: int | None = None) -> pd.DataFrame:
    """Adjusted daily bars for a list of tickers, long format, cached."""
    start = start or config.HISTORY_START
    ttl = config.CACHE_TTL_SECONDS if ttl is None else ttl
    key = f"{start}|{','.join(sorted(tickers))}"

    hit = _cached(key, ttl)
    if hit is not None:
        log.info("prices: %d rows from cache", len(hit))
        return hit

    errors = []
    for name, fn in (("yfinance", _from_yfinance), ("stooq", _from_stooq)):
        try:
            df = fn(tickers, start)
        except (PricesUnavailable, ImportError) as exc:
            errors.append(f"{name}: {exc}")
            log.warning("%s unusable: %s", name, exc)
            continue
        df = (df.sort_values(["ticker", "date"])
                .drop_duplicates(["ticker", "date"], keep="last")
                .reset_index(drop=True))
        df["source"] = name
        _store(key, df)
        log.info("prices: %d rows, %d tickers, %s to %s (%s)",
                 len(df), df["ticker"].nunique(),
                 df["date"].min().date(), df["date"].max().date(), name)
        return df

    raise PricesUnavailable("; ".join(errors))


# ------------------------------------------------------- returns and vol
def add_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, per ticker.

    Log returns because they add across time, which is what makes a horizon
    return the sum of its days and lets realized volatility scale by the square
    root of the window.
    """
    df = px.sort_values(["ticker", "date"]).copy()
    df["ret"] = np.log(df["close"] / df.groupby("ticker")["close"].shift(1))
    df["abs_ret"] = df["ret"].abs()
    df["gap"] = np.log(df["open"] / df.groupby("ticker")["close"].shift(1))
    # Parkinson's estimator: the day's own high-low range carries information
    # about volatility that a close-to-close return throws away.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["range_vol"] = np.log(df["high"] / df["low"]) / np.sqrt(4 * np.log(2))
    return df


def realized_vol(df: pd.DataFrame, windows: list[int] | None = None
                 ) -> pd.DataFrame:
    """Trailing realized volatility, annualised, per ticker.

    Strictly trailing: every window ends at the row it labels, so nothing here
    can see a return from the day it is used to predict. `min_periods` is set
    high enough that a half-filled window does not masquerade as a measurement.
    """
    windows = windows or config.RV_WINDOWS
    out = df.sort_values(["ticker", "date"]).copy()
    g = out.groupby("ticker")["ret"]
    for w in windows:
        out[f"rv_{w}"] = (g.rolling(w, min_periods=max(3, w // 2)).std()
                           .reset_index(level=0, drop=True) * np.sqrt(252))
    # The shape of the term structure: a stock whose short vol sits far above
    # its long vol is in something, and that something usually persists.
    if len(windows) >= 2:
        short, long = f"rv_{windows[0]}", f"rv_{windows[-1]}"
        out["rv_slope"] = out[short] / out[long].replace(0, np.nan)
    return out


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Log return from each row's close to the close `horizon` days later.

    This is the target. It is the one column in the project that looks into the
    future, which is why it is computed in exactly one place and named plainly.
    """
    g = df.sort_values(["ticker", "date"]).groupby("ticker")["close"]
    return np.log(g.shift(-horizon) / g.shift(0))


def audit(df: pd.DataFrame) -> dict:
    """Evidence that the adjustment actually happened.

    An unadjusted 2-for-1 split reads as a 50% single-day loss. A handful of
    those in a decade is real news; hundreds means the source is not adjusting
    and every volatility number downstream is inflated.
    """
    r = df["ret"].dropna()
    huge = r.abs() > config.IMPLAUSIBLE_DAILY_MOVE
    near_half = (r.between(-0.75, -0.63)) | (r.between(0.63, 0.75))
    return {
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "returns": int(len(r)),
        "daily_sd": float(r.std()),
        "implausible_moves": int(huge.sum()),
        "implausible_share": float(huge.mean()) if len(r) else float("nan"),
        # A cluster right at log(2) is the signature of unadjusted splits.
        "split_shaped_moves": int(near_half.sum()),
        "max_abs_move": float(r.abs().max()) if len(r) else float("nan"),
    }
