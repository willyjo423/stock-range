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
import re
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
    return _members_from_blob(_get(config.SP500_CURRENT_URLS))


def _members_from_blob(blob: bytes) -> list[str]:
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


_TICKERISH = re.compile(r"^[A-Z][A-Z0-9.\-]{0,5}$")
# Enough rows to be the change log rather than an infobox that mentions a date.
_MIN_CHANGE_ROWS = 10


def _flatten(columns) -> list[str]:
    return [" ".join(str(x) for x in c).lower() if isinstance(c, tuple)
            else str(c).lower() for c in columns]


def _date_like(s: pd.Series) -> int:
    """How many entries in this column parse as dates."""
    for kwargs in ({"format": "mixed"}, {}):
        try:
            return int(pd.to_datetime(s, errors="coerce",
                                      **kwargs).notna().sum())
        except (ValueError, TypeError):
            continue
    return 0


def _ticker_like(s: pd.Series) -> float:
    """What share of a column looks like a ticker symbol.

    This is how the added and removed columns get found without trusting the
    header, because the header is the part of a scraped page most likely to be
    renamed. A column of company names or a sentence of explanation scores near
    zero; a column of NVDA and POOL scores near one.
    """
    vals = [str(v).strip() for v in s.dropna().tolist()]
    vals = [v for v in vals if v and v.lower() not in ("nan", "-", "—")]
    if len(vals) < 5:
        return 0.0
    return sum(bool(_TICKERISH.match(v)) for v in vals) / len(vals)


def _tidy(out: pd.DataFrame) -> pd.DataFrame:
    for kwargs in ({"format": "mixed"}, {}):
        try:
            out["date"] = pd.to_datetime(out["date"], errors="coerce", **kwargs)
            break
        except (ValueError, TypeError):
            continue
    return (out.dropna(subset=["date"]).sort_values("date")
               .reset_index(drop=True))


def _inventory(tables: list[pd.DataFrame]) -> str:
    """What was actually on the page, for an error message worth reading.

    A scraper that fails with "not found" sends whoever reads it back to the
    website to guess. Printing the shape and headers of every table found turns
    the next run into the diagnosis rather than into the next attempt.
    """
    bits = []
    for i, t in enumerate(tables[:12]):
        cols = ", ".join(_flatten(t.columns))[:110]
        bits.append(f"[{i}] {len(t)}x{len(t.columns)} ({cols})")
    if len(tables) > 12:
        bits.append(f"...and {len(tables) - 12} more")
    return " | ".join(bits)


def _changes_table(blob: bytes) -> pd.DataFrame:
    """The 'Selected changes to the list' table: date, added, removed.

    Two strategies, because this is a scrape of a page that gets restructured
    and the first live run after it was restructured found nothing.

    **By header** - a column mentioning "added" and one mentioning "removed".
    Unambiguous when it holds, which it did until the page grew a two-row
    header of Added/Removed over Ticker/Security.

    **By shape** - a column that parses as dates and two columns full of things
    that look like ticker symbols, in that order. This survives any renaming.
    The added column has preceded the removed one in every layout this table
    has had; if that ever reverses, the count check downstream catches it
    immediately, because undoing changes in the wrong direction makes the
    membership count drift away from 500 within a few years.

    If both fail it raises with an inventory of every table on the page, so the
    next run diagnoses the problem instead of repeating it.
    """
    tables = pd.read_html(io.BytesIO(blob))
    candidates = [t for t in tables if len(t) >= _MIN_CHANGE_ROWS]

    # --- by header ---------------------------------------------------------
    best = None
    for tbl in candidates:
        flat = _flatten(tbl.columns)
        if (any("date" in c for c in flat) and any("added" in c for c in flat)
                and any("removed" in c for c in flat)):
            if best is None or len(tbl) > len(best):
                best = tbl.copy()
                best.columns = flat

    if best is not None:
        def _pick(*needles):
            for c in best.columns:
                if all(n in c for n in needles):
                    return c
            return None

        date_c = _pick("date")
        add_c = (_pick("added", "ticker") or _pick("added", "symbol")
                 or _pick("added"))
        rem_c = (_pick("removed", "ticker") or _pick("removed", "symbol")
                 or _pick("removed"))
        if date_c and add_c and rem_c and add_c != rem_c:
            out = best[[date_c, add_c, rem_c]].copy()
            out.columns = ["date", "added", "removed"]
            parsed = _tidy(out)
            if len(parsed) >= _MIN_CHANGE_ROWS:
                log.info("changes table matched by header: %d changes",
                         len(parsed))
                return parsed
        log.warning("a table carried added/removed headers but did not "
                    "parse: %s", list(best.columns))

    # --- by shape ----------------------------------------------------------
    for tbl in sorted(candidates, key=len, reverse=True):
        cols = list(tbl.columns)
        scored = [(c, _date_like(tbl[c])) for c in cols]
        date_c, hits = max(scored, key=lambda p: p[1], default=(None, 0))
        if hits < _MIN_CHANGE_ROWS:
            continue
        tickers = [c for c in cols
                   if c is not date_c and _ticker_like(tbl[c]) >= 0.6]
        if len(tickers) < 2:
            continue
        out = tbl[[date_c, tickers[0], tickers[1]]].copy()
        out.columns = ["date", "added", "removed"]
        parsed = _tidy(out)
        if len(parsed) >= _MIN_CHANGE_ROWS:
            log.warning("changes table matched by shape rather than by "
                        "header - the page layout has changed. Columns used: "
                        "%s", [str(date_c), str(tickers[0]),
                               str(tickers[1])])
            return parsed

    raise UniverseUnavailable(
        f"no changes table on the page. {len(tables)} tables found: "
        f"{_inventory(tables)}")


def reconstruct_members(current: list[str], blob: bytes) -> pd.DataFrame:
    """Walk today's index backwards through the change log.

    Every row of that table says "X came in, Y went out, on this date". Playing
    it in reverse from today rebuilds the membership on any past date: undoing
    a change means removing what was added and restoring what was removed.

    This is reconstruction, not a record, so it is only as complete as the
    change log. That is why `Universe` checks the resulting count stays near
    500 throughout - if changes are being missed, the count drifts, and a
    drifting count is the tell.
    """
    changes = _changes_table(blob)
    members = set(current)
    rows = []

    for r in changes.iloc[::-1].itertuples(index=False):
        # `members` currently holds the membership that applied from this
        # change date onward, so record it against that date *before* undoing
        # it. Recording after would label each row with the date the state
        # stopped being true rather than the date it started.
        rows.append({"date": pd.Timestamp(r.date), "members": sorted(members),
                     "n": len(members)})
        added = normalise_ticker(r.added) if pd.notna(r.added) else ""
        removed = normalise_ticker(r.removed) if pd.notna(r.removed) else ""
        if added and added in members:
            members.discard(added)
        if removed:
            members.add(removed)

    # Whatever is left is the membership before the earliest change we know of.
    earliest = (pd.Timestamp(changes["date"].min()) - pd.Timedelta(days=1)
                if len(changes) else pd.Timestamp(config.HISTORY_START))
    rows.append({"date": earliest, "members": sorted(members),
                 "n": len(members)})

    df = (pd.DataFrame(rows).sort_values("date")
            .drop_duplicates("date", keep="last").reset_index(drop=True))
    log.info("reconstructed membership from %d changes: %s to %s, "
             "count %d-%d", len(changes), df["date"].min().date(),
             df["date"].max().date(), df["n"].min(), df["n"].max())
    return df[["date", "members", "n"]]


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
        blob = None
        try:
            blob = _get(config.SP500_CURRENT_URLS)
            current = _members_from_blob(blob)
        except UniverseUnavailable as exc:
            log.error("current membership unavailable: %s", exc)
            current = []

        published, rebuilt = None, None
        try:
            published = historical_members()
        except UniverseUnavailable as exc:
            log.warning("membership history file unavailable: %s", exc)
        if blob is not None and current:
            try:
                rebuilt = reconstruct_members(current, blob)
            except UniverseUnavailable as exc:
                log.warning("reconstruction failed: %s", exc)

        # The published file is a record and the reconstruction is an
        # inference, so the file wins where it exists. But the copy that
        # answered on the first live run stops in January 2019 - seven years
        # short - and using it alone would freeze membership there and lose
        # every name that has joined and left since. So the two are spliced:
        # the file for the deep history, the change log for everything after
        # it ends.
        history, source = None, None
        if published is not None and rebuilt is not None:
            cutoff = pd.Timestamp(published["date"].max())
            recent = rebuilt[rebuilt["date"] > cutoff]
            history = (pd.concat([published, recent], ignore_index=True)
                         .sort_values("date").reset_index(drop=True))
            source = (f"published file to {cutoff.date()}, then the change log "
                      f"({len(recent)} later changes)")
        elif published is not None:
            history, source = published, "published file only"
            log.warning("no change log to extend the file past %s; membership "
                        "is frozen after that date",
                        published["date"].max().date())
        elif rebuilt is not None:
            history, source = rebuilt, "reconstructed from the change log"

        if not current and history is not None and len(history):
            current = list(history.iloc[-1]["members"])
        if not current:
            raise UniverseUnavailable("no membership from any source")

        uni = cls(history, current)
        uni.source = source
        return uni

    # Set by `load`; describes where the history came from.
    source: str | None = None

    def count_check(self) -> dict:
        """Does the membership count stay near 500 throughout?

        The index has held roughly 500 names for decades, so a reconstruction
        that drifts to 380 or 620 is missing or double-applying changes. This
        is the cheapest possible validation and it needs no second source.
        """
        if not self.survivorship_safe:
            return {"ok": False, "reason": "no history"}
        lo, hi = config.MEMBERSHIP_COUNT_BAND
        n = self.history["n"]
        in_band = float(((n >= lo) & (n <= hi)).mean())
        return {"ok": in_band > 0.95, "min": int(n.min()), "max": int(n.max()),
                "median": int(n.median()), "share_in_band": in_band,
                "band": (lo, hi)}

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
