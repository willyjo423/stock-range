"""Who was in the index, and when.

This file exists because of one bias that would otherwise quietly flatter every
number the model produces.

If you take today's S&P 500 and run it back twenty years, you have selected for
companies that survived. The ones that blew up, got acquired in distress, or
were dropped for being too small are all missing - and those are exactly the
names whose volatility went through the roof before they left. A model trained
on the survivors learns that stocks are calmer than they are, and it will be
most wrong precisely when it matters.

So membership is treated as a function of date, not a fixed list. When the
historical file cannot be fetched the loader still works, but it says loudly
that it is running survivorship-biased and the probe repeats the warning.
"""
from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd
import requests

import config

log = logging.getLogger(__name__)


class UniverseUnavailable(RuntimeError):
    """Neither current membership nor history could be fetched."""


def _get(urls: list[str]) -> bytes:
    last = None
    for url in urls:
        try:
            resp = requests.get(
                url, timeout=config.REQUEST_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0 (range-forecast research)"})
            if resp.status_code == 200 and resp.content:
                return resp.content
            last = f"HTTP {resp.status_code} from {url}"
        except requests.RequestException as exc:
            last = f"{url}: {exc}"
        log.warning("universe fetch failed: %s", last)
    raise UniverseUnavailable(str(last))


def normalise_ticker(t: str) -> str:
    """Yahoo spells class shares with a dash where the index uses a dot.

    BRK.B is BRK-B, BF.B is BF-B. Getting this wrong silently drops two of the
    larger names in the index rather than raising anything.
    """
    return str(t).strip().upper().replace(".", "-")


def current_members() -> list[str]:
    """Today's index, from the Wikipedia table."""
    blob = _get(config.SP500_CURRENT_URLS)
    tables = pd.read_html(io.BytesIO(blob))
    for tbl in tables:
        cols = {str(c).strip().lower() for c in tbl.columns}
        if "symbol" in cols:
            col = [c for c in tbl.columns if str(c).strip().lower() == "symbol"][0]
            out = sorted({normalise_ticker(s) for s in tbl[col].dropna()})
            if len(out) > 400:
                log.info("current S&P 500 membership: %d tickers", len(out))
                return out
    raise UniverseUnavailable("no table on that page had a Symbol column")


def historical_members() -> pd.DataFrame:
    """One row per date, with the full membership list as of that date.

    The source publishes a row each time the index changed, so the result is a
    step function: look up the latest row on or before your date.
    """
    blob = _get(config.SP500_HISTORY_URLS)
    df = pd.read_csv(io.BytesIO(blob))

    date_col = next((c for c in df.columns if "date" in str(c).lower()), None)
    tick_col = next((c for c in df.columns
                     if "ticker" in str(c).lower() or "symbol" in str(c).lower()),
                    None)
    if date_col is None or tick_col is None:
        raise UniverseUnavailable(f"unexpected columns: {list(df.columns)}")

    df = df[[date_col, tick_col]].rename(
        columns={date_col: "date", tick_col: "tickers"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["members"] = df["tickers"].astype(str).apply(
        lambda s: sorted({normalise_ticker(x) for x in s.split(",") if x.strip()}))
    df["n"] = df["members"].apply(len)
    log.info("membership history: %d change dates, %s to %s",
             len(df), df["date"].min().date(), df["date"].max().date())
    return df[["date", "members", "n"]]


class Universe:
    """Index membership as of any date, with an honest fallback."""

    def __init__(self, history: pd.DataFrame | None, current: list[str]):
        self.history = history
        self.current = current
        self.survivorship_safe = history is not None and not history.empty
        if not self.survivorship_safe:
            log.warning(
                "NO MEMBERSHIP HISTORY - falling back to today's index for all "
                "dates. Every backtest from here is survivorship-biased and "
                "will understate volatility, because the companies that blew "
                "up are the ones missing.")

    @classmethod
    def load(cls) -> "Universe":
        try:
            current = current_members()
        except UniverseUnavailable as exc:
            log.error("current membership unavailable: %s", exc)
            current = []
        try:
            history = historical_members()
        except UniverseUnavailable as exc:
            log.warning("membership history unavailable: %s", exc)
            history = None
        if not current and history is not None and len(history):
            current = list(history.iloc[-1]["members"])
        if not current:
            raise UniverseUnavailable("no membership from any source")
        return cls(history, current)

    def members_on(self, when) -> list[str]:
        if not self.survivorship_safe:
            return list(self.current)
        when = pd.Timestamp(when)
        rows = self.history[self.history["date"] <= when]
        if rows.empty:
            return list(self.history.iloc[0]["members"])
        return list(rows.iloc[-1]["members"])

    def all_ever(self) -> list[str]:
        """Every ticker that was ever a member in the window we model.

        This is what gets downloaded. It is meaningfully larger than 500 -
        roughly a thousand once two decades of turnover are included - and the
        difference between the two numbers is the size of the bias avoided.
        """
        if not self.survivorship_safe:
            return list(self.current)
        start = pd.Timestamp(config.HISTORY_START)
        seen: set[str] = set()
        for _, row in self.history.iterrows():
            if row["date"] >= start:
                seen.update(row["members"])
        seen.update(self.members_on(start))
        seen.update(self.current)
        return sorted(seen)

    def summary(self) -> dict:
        ever = self.all_ever()
        return {
            "survivorship_safe": self.survivorship_safe,
            "current": len(self.current),
            "ever_in_window": len(ever),
            "turnover": len(ever) - len(self.current),
            "change_dates": 0 if self.history is None else int(len(self.history)),
        }
