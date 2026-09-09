"""Earnings dates, fetched once and cached hard.

The probe found these go back a median of 12.3 years, which is deep enough to
train on. Getting them is the awkward part: there is no bulk endpoint, so it is
one request per ticker, and a thousand tickers at a second each is a twenty
minute wait before any modelling starts.

So the same discipline the football build's weather service ended up with: a
permanent disk cache, a hard time budget, and a failure that costs one feature
rather than the whole run. A ticker that cannot be fetched keeps NaN dates,
which the booster handles natively - and NaN is the honest answer, because "no
earnings in this window" and "we do not know" are different claims.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

import config

log = logging.getLogger(__name__)

CACHE = config.CACHE / "earnings.pkl.gz"
DEFAULT_BUDGET_SECONDS = 1800.0


def _load_cache() -> pd.DataFrame:
    if CACHE.exists():
        try:
            return pd.read_pickle(CACHE, compression="gzip")
        except Exception as exc:  # noqa: BLE001 - a bad cache is not fatal
            log.warning("unreadable earnings cache: %s", exc)
    return pd.DataFrame(columns=["ticker", "date"])


def _save_cache(df: pd.DataFrame) -> None:
    try:
        df.to_pickle(CACHE, compression="gzip")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not write earnings cache: %s", exc)


def load(tickers: list[str], budget_seconds: float = DEFAULT_BUDGET_SECONDS,
         refresh: bool = False) -> pd.DataFrame:
    """Report dates for as many tickers as the budget allows."""
    cached = _load_cache()
    have = set(cached["ticker"].unique()) if len(cached) else set()
    todo = [t for t in tickers if refresh or t not in have]

    if not todo:
        log.info("earnings: %d tickers from cache", len(have & set(tickers)))
        return cached[cached["ticker"].isin(tickers)]

    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance unavailable; earnings features will be empty")
        return cached[cached["ticker"].isin(tickers)]

    started = time.time()
    rows, done, failed = [], 0, 0
    for t in todo:
        if time.time() - started > budget_seconds:
            log.warning("earnings budget spent; %d tickers unfetched",
                        len(todo) - done)
            break
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=80)
        except Exception:  # noqa: BLE001 - one ticker, not the run
            failed += 1
            done += 1
            continue
        done += 1
        if df is None or len(df) == 0:
            continue
        idx = pd.to_datetime(pd.Series(df.index), errors="coerce", utc=True)
        idx = idx.dropna().dt.tz_localize(None)
        rows.extend({"ticker": t, "date": d} for d in idx)
        if done % 100 == 0:
            log.info("earnings: %d/%d tickers (%d failed)", done, len(todo),
                     failed)

    fresh = pd.DataFrame(rows)
    out = (pd.concat([cached, fresh], ignore_index=True)
           if len(fresh) else cached)
    if len(out):
        out = out.drop_duplicates(["ticker", "date"]).reset_index(drop=True)
        _save_cache(out)
    log.info("earnings: %d dates across %d tickers (%d failed to fetch)",
             len(out), out["ticker"].nunique() if len(out) else 0, failed)
    return out[out["ticker"].isin(tickers)] if len(out) else out


def coverage(earnings: pd.DataFrame, tickers: list[str]) -> dict:
    if earnings is None or earnings.empty:
        return {"tickers_with_dates": 0, "share": 0.0, "dates": 0}
    have = earnings["ticker"].nunique()
    return {
        "tickers_with_dates": int(have),
        "share": float(have / max(len(tickers), 1)),
        "dates": int(len(earnings)),
        "earliest": str(earnings["date"].min().date()),
        "latest": str(earnings["date"].max().date()),
    }
