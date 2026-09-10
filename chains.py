"""Option chains from free sources, normalised to one tidy frame.

Two sources, tried in order, for the same reason the price loader has two:

* **CBOE delayed quotes** - one JSON document per underlying containing every
  expiry, with volume, open interest, quotes and greeks. One request per name.
* **yfinance** - the same information, but one request per expiry, so five to
  ten requests per name. Fine for a handful of tickers, too slow for five
  hundred inside a schedule slot.

Both are delayed by roughly fifteen minutes. That is not a problem for what
this is used for: nothing here is a trade signal that decays in seconds, and
the flags get graded over the following week.

What neither gives, and no free source gives, is the tape - the individual
prints with size and the quote standing at the moment of each one. Everything
downstream is built on the snapshot and is careful to say so.

One property of the snapshot worth stating loudly because it is load-bearing:
**open interest does not tick intraday.** The figure in a chain pulled at 11am
is last night's settled open interest. So comparing today's volume against it
is a clean before-and-after, not a circular comparison of a number against
itself. If a future source ever starts updating OI live, that assumption
breaks and `flow.py` starts measuring nothing.
"""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config

log = logging.getLogger(__name__)

COLUMNS = ["ticker", "contract", "expiry", "right", "strike", "bid", "ask",
           "last", "volume", "open_interest", "iv", "spot", "snapshot_at",
           "source"]


class ChainsUnavailable(RuntimeError):
    """No source returned a usable chain."""


# ------------------------------------------------------------ OCC symbols
# AAPL  240920 C 00220000  ->  root, expiry, right, strike x 1000
_OCC = re.compile(r"^(?P<root>[A-Z0-9\.\-]{1,6})"
                  r"(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})"
                  r"(?P<right>[CP])"
                  r"(?P<strike>\d{8})$")


def parse_occ(symbol: str) -> dict | None:
    """Pull expiry, right and strike out of an OCC contract symbol.

    Used rather than trusting a separate expiry field, because the contract
    symbol is the one thing every source agrees on and it cannot disagree with
    itself. A chain whose stated expiry and whose symbol disagree is a chain
    with a bug in it, and `check()` reports that rather than picking one.
    """
    m = _OCC.match(str(symbol).strip().upper())
    if not m:
        return None
    try:
        expiry = pd.Timestamp(year=2000 + int(m["y"]), month=int(m["m"]),
                              day=int(m["d"]))
    except ValueError:
        return None
    return {"root": m["root"], "expiry": expiry, "right": m["right"],
            "strike": int(m["strike"]) / 1000.0}


# ------------------------------------------------------------------- CBOE
def _num(d: dict, *names, default=np.nan) -> float:
    """First of several possible key spellings that holds a number.

    The free endpoints rename fields between revisions and the failure mode of
    guessing wrong is a column of NaN that looks like a quiet market rather
    than like a broken parser. Trying the known spellings and reporting what
    was found is cheaper than discovering it downstream.
    """
    for n in names:
        if n in d and d[n] is not None:
            try:
                v = float(d[n])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                return v
    return default


def _cboe_urls(ticker: str) -> list[str]:
    sym = ticker.upper().replace("-", "")
    urls = [config.CBOE_CHAIN_URL.format(sym=sym)]
    if ticker.startswith("^"):
        urls = [config.CBOE_INDEX_URL.format(sym=ticker[1:].upper())]
    return urls


def from_cboe(ticker: str, session: requests.Session | None = None
              ) -> pd.DataFrame:
    """One underlying's whole chain, from the delayed-quote JSON."""
    sess = session or requests
    last_exc: Exception | None = None
    for url in _cboe_urls(ticker):
        try:
            r = sess.get(url, timeout=config.REQUEST_TIMEOUT,
                         headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                last_exc = ChainsUnavailable(f"{url} -> HTTP {r.status_code}")
                continue
            payload = r.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_exc = exc
            continue

        data = payload.get("data") or payload
        rows = data.get("options") or data.get("option") or []
        if not rows:
            last_exc = ChainsUnavailable(f"{url} -> no options array")
            continue

        spot = _num(data, "current_price", "close", "last", "prev_day_close")
        stamp = pd.Timestamp(datetime.now(timezone.utc)).floor("min")

        out = []
        for row in rows:
            sym = row.get("option") or row.get("symbol") or row.get("contract")
            meta = parse_occ(sym or "")
            if not meta:
                continue
            out.append({
                "ticker": ticker,
                "contract": sym,
                "expiry": meta["expiry"],
                "right": meta["right"],
                "strike": meta["strike"],
                "bid": _num(row, "bid"),
                "ask": _num(row, "ask"),
                "last": _num(row, "last_trade_price", "last_price", "last"),
                "volume": _num(row, "volume", default=0.0),
                "open_interest": _num(row, "open_interest", "openInterest",
                                      default=np.nan),
                "iv": _num(row, "iv", "implied_volatility",
                           "impliedVolatility"),
                "spot": spot,
                "snapshot_at": stamp,
                "source": "cboe",
            })
        if out:
            return pd.DataFrame(out, columns=COLUMNS)
        last_exc = ChainsUnavailable(f"{url} -> no parseable contracts")

    raise ChainsUnavailable(f"CBOE: {ticker}: {last_exc}")


# --------------------------------------------------------------- yfinance
def from_yfinance(ticker: str, max_dte: int | None = None) -> pd.DataFrame:
    """The fallback. Correct, and one request per expiry."""
    import yfinance as yf

    max_dte = config.FLOW_MAX_DTE if max_dte is None else max_dte
    t = yf.Ticker(ticker)
    try:
        expiries = list(t.options or [])
    except Exception as exc:  # noqa: BLE001 - any failure means "no chain"
        raise ChainsUnavailable(f"yfinance: {ticker}: {exc}") from exc
    if not expiries:
        raise ChainsUnavailable(f"yfinance: {ticker}: no expirations listed")

    today = pd.Timestamp.today().normalize()
    wanted = [e for e in expiries
              if 0 <= (pd.Timestamp(e) - today).days <= max_dte + 3]
    if not wanted:
        wanted = expiries[:2]

    spot = np.nan
    try:
        hist = t.history(period="5d", auto_adjust=False)
        if hist is not None and not hist.empty:
            spot = float(hist["Close"].dropna().iloc[-1])
    except Exception as exc:  # noqa: BLE001
        log.debug("no spot for %s: %s", ticker, exc)

    stamp = pd.Timestamp(datetime.now(timezone.utc)).floor("min")
    frames = []
    for exp in wanted:
        try:
            chain = t.option_chain(exp)
        except Exception as exc:  # noqa: BLE001
            log.debug("%s %s: %s", ticker, exp, exc)
            continue
        for side, df in (("C", chain.calls), ("P", chain.puts)):
            if df is None or df.empty:
                continue
            sub = pd.DataFrame({
                "ticker": ticker,
                "contract": df.get("contractSymbol"),
                "expiry": pd.Timestamp(exp),
                "right": side,
                "strike": pd.to_numeric(df.get("strike"), errors="coerce"),
                "bid": pd.to_numeric(df.get("bid"), errors="coerce"),
                "ask": pd.to_numeric(df.get("ask"), errors="coerce"),
                "last": pd.to_numeric(df.get("lastPrice"), errors="coerce"),
                "volume": pd.to_numeric(df.get("volume"),
                                        errors="coerce").fillna(0.0),
                "open_interest": pd.to_numeric(df.get("openInterest"),
                                               errors="coerce"),
                "iv": pd.to_numeric(df.get("impliedVolatility"),
                                    errors="coerce"),
                "spot": spot,
                "snapshot_at": stamp,
                "source": "yfinance",
            })
            frames.append(sub)

    if not frames:
        raise ChainsUnavailable(f"yfinance: {ticker}: no chain rows")
    return pd.concat(frames, ignore_index=True)[COLUMNS]


# ------------------------------------------------------------------ fetch
def fetch(ticker: str, prefer: str = "cboe",
          session: requests.Session | None = None) -> pd.DataFrame:
    """One ticker, whichever source answers."""
    order = ["cboe", "yfinance"] if prefer == "cboe" else ["yfinance", "cboe"]
    errors = []
    for name in order:
        try:
            df = (from_cboe(ticker, session=session) if name == "cboe"
                  else from_yfinance(ticker))
            if not df.empty:
                return df
            errors.append(f"{name}: empty")
        except ChainsUnavailable as exc:
            errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise ChainsUnavailable("; ".join(errors))


def fetch_many(tickers: list[str], prefer: str = "cboe",
               max_workers: int | None = None,
               on_error: str = "skip") -> tuple[pd.DataFrame, dict[str, str]]:
    """Every ticker's chain, in parallel, with the failures kept.

    Failures are returned rather than logged and forgotten. A scan that quietly
    covered 310 of 500 names looks identical on the page to one that covered
    all of them, and the difference is the whole reason to trust the count.
    """
    workers = max_workers or config.FLOW_MAX_WORKERS
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    session = requests.Session()

    def one(t: str):
        time.sleep(config.FLOW_REQUEST_PAUSE)
        return t, fetch(t, prefer=prefer, session=session)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            t = futures[fut]
            done += 1
            try:
                _, df = fut.result()
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                errors[t] = str(exc)[:200]
                if on_error == "raise":
                    raise
            if done % 50 == 0:
                log.info("chains: %d/%d (%d failed)", done, len(tickers),
                         len(errors))

    if not frames:
        raise ChainsUnavailable(
            f"no chains returned for any of {len(tickers)} tickers; "
            f"first error: {next(iter(errors.values()), 'none')}")
    out = pd.concat(frames, ignore_index=True)
    log.info("chains: %d contracts across %d tickers, %d failed",
             len(out), out["ticker"].nunique(), len(errors))
    return out, errors


# ------------------------------------------------------------------ check
def check(df: pd.DataFrame) -> dict:
    """What a chain actually contains, as numbers rather than as faith.

    Run by the probe and by every scan. The three that matter:

    `symbol_disagreements` - rows whose stated expiry or strike contradicts
    their own contract symbol. Any at all means the parser is wrong about that
    source and nothing downstream should be believed.

    `oi_missing` / `volume_missing` - the two fields the whole scan rests on.
    A source that returns quotes but no open interest is useless here, and it
    is better to learn that in one line than from an empty page.

    `zero_volume_share` - the ordinary state of an option chain is that almost
    nothing trades. If this is low, something is being double-counted.
    """
    if df is None or df.empty:
        return {"rows": 0, "usable": False}

    disagree = 0
    for sym, exp, k in zip(df["contract"], df["expiry"], df["strike"]):
        meta = parse_occ(sym or "")
        if not meta:
            continue
        if pd.notna(exp) and pd.Timestamp(exp).normalize() != meta["expiry"]:
            disagree += 1
        elif pd.notna(k) and abs(float(k) - meta["strike"]) > 1e-6:
            disagree += 1

    vol = pd.to_numeric(df["volume"], errors="coerce")
    oi = pd.to_numeric(df["open_interest"], errors="coerce")
    return {
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "sources": sorted(set(df["source"].dropna())),
        "expiries": int(df["expiry"].nunique()),
        "symbol_disagreements": int(disagree),
        "oi_missing": float(oi.isna().mean()),
        "volume_missing": float(vol.isna().mean()),
        "spot_missing": float(pd.to_numeric(df["spot"],
                                            errors="coerce").isna().mean()),
        "zero_volume_share": float((vol.fillna(0) <= 0).mean()),
        "usable": bool(disagree == 0 and oi.isna().mean() < 0.5
                       and vol.isna().mean() < 0.5),
    }
