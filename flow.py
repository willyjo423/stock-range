"""The options-flow scan: what free data can and cannot say about sharp action.

The target is a specific pattern - short-dated, at the money, large premium,
into a contract that barely existed yesterday. On a real tape that is one line
you can point at. Here it has to be reconstructed from snapshots, and the
reconstruction is honest about which parts survive and which do not.

What survives intact
--------------------
**Days to expiry** and **moneyness** are properties of the contract and the
spot price. Exact.

**Open interest going in.** Open interest settles overnight and does not tick
during the session, so the figure in any intraday snapshot is yesterday's.
Comparing today's volume against it is a genuine before-and-after. This is the
strongest free filter by some distance, and it is the one closest to what a
flow tool means by "new positioning".

What degrades
-------------
**Premium.** On a tape this is one trade's size times its price. Here it is
`volume x mid x 100` - all the premium that traded in that contract, however
many hands it took. One $80,000 sweep and eight hundred $100 retail lots are
the same number.

The only free lever against that is time. Scan several times a session and
difference the cumulative volume: a block lands almost entirely inside one
interval, a dribble spreads across all of them. `burst` is that share, and a
burst near 1.0 on a contract with meaningful premium is the closest this can
get to "one trade did that". It needs at least two snapshots in the day and
reports `None` until it has them, rather than defaulting to something
flattering.

**Side.** Whether the premium was bought or sold needs the quote at the instant
of the trade. What is available is the last trade price against the current
quote, which is a decent proxy when the trade just happened and meaningless
when it happened three hours ago. So the proxy is computed, and it is marked
`firm` only when the interval's own volume shows the trade fell inside it.

What is not attempted
---------------------
No attempt to identify the opening or closing intent of any trade, which was
dropped from the brief; no attempt to detect sweeps, exchanges, or whether the
order was part of a spread beyond one crude same-expiry check. Those need the
tape.

And the whole thing is gradeable - see `flow_grade.py`. Every flag is archived
with the date it was raised, and graded a week later against the range model's
own band for that stock. A scan that cannot be checked is a horoscope.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import chains
import config

log = logging.getLogger(__name__)

# Contracts kept in the stored snapshot. Wider than the ATM band the screen
# uses, because a snapshot is also the previous snapshot for the next scan and
# a spot move during the day shifts which strikes are at the money.
STORE_MONEYNESS = 0.15
# Snapshots older than this are deleted. They exist to be differenced within a
# session and to answer "did open interest confirm this the next morning"; a
# fortnight covers both with room to spare.
SNAPSHOT_RETENTION_DAYS = 14


# --------------------------------------------------------------- enrichment
def enrich(df: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Everything derivable from a single snapshot, and nothing more."""
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["mid", "dte"])
    out = df.copy()
    asof = pd.Timestamp(asof).normalize() if asof is not None else \
        pd.Timestamp(out["snapshot_at"].max()).tz_localize(None).normalize()

    for c in ("bid", "ask", "last", "volume", "open_interest", "strike",
              "spot", "iv"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["expiry"] = pd.to_datetime(out["expiry"]).dt.normalize()
    out["dte"] = (out["expiry"] - asof).dt.days

    bid, ask = out["bid"], out["ask"]
    crossed = (ask < bid) | (ask <= 0)
    mid = (bid + ask) / 2.0
    # A missing or crossed two-sided market is not a mid. Falling back to the
    # last trade is the least-bad option and is marked, because premium built
    # on a stale last is the main way a number here can be wrong by 10x.
    out["mid"] = np.where(mid.notna() & ~crossed & (mid > 0), mid, out["last"])
    out["mid_from_last"] = (mid.isna() | crossed | (mid <= 0)).astype(int)
    out["spread_pct"] = np.where(out["mid"] > 0, (ask - bid) / out["mid"],
                                 np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_moneyness"] = np.log(out["strike"] / out["spot"])
    out["volume"] = out["volume"].fillna(0.0)
    out["day_premium"] = out["volume"] * out["mid"] * 100.0
    oi = out["open_interest"]
    out["vol_oi"] = out["volume"] / oi.clip(lower=1.0)

    # Where the last print sits inside the quoted market. +1 is at the offer,
    # -1 at the bid. Only meaningful if the print is recent, which this frame
    # cannot know - `interval()` is what decides that.
    half = (ask - bid) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        aggr = (out["last"] - mid) / half
    out["side_proxy"] = np.clip(aggr, -1.0, 1.0)
    return out


def interval(now: pd.DataFrame, prev: pd.DataFrame | None) -> pd.DataFrame:
    """Volume that arrived since the previous snapshot.

    Cumulative day volume only rises, so the difference is new trading. It is
    clipped at zero because a source occasionally revises a figure downward and
    a negative "new volume" is noise, not a sale.

    With no previous snapshot the interval is the whole day so far, and the
    columns say so: `interval_is_day` is 1 and `burst` is None rather than 1.0,
    which would claim concentration that was never measured.
    """
    out = now.copy()
    if prev is None or prev.empty:
        out["new_volume"] = out["volume"]
        out["interval_is_day"] = 1
        out["burst"] = np.nan
        out["prev_volume"] = np.nan
    else:
        key = ["ticker", "contract"]
        before = (prev[key + ["volume"]]
                  .rename(columns={"volume": "prev_volume"})
                  .drop_duplicates(key))
        out = out.merge(before, on=key, how="left")
        out["new_volume"] = (out["volume"] -
                             out["prev_volume"].fillna(0.0)).clip(lower=0.0)
        out["interval_is_day"] = out["prev_volume"].isna().astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["burst"] = np.where(out["volume"] > 0,
                                    out["new_volume"] / out["volume"], np.nan)
        out.loc[out["interval_is_day"] == 1, "burst"] = np.nan

    out["new_premium"] = out["new_volume"] * out["mid"] * 100.0
    # The side proxy is only worth reading when the print fell inside the
    # window just measured. Otherwise `last` may be hours stale.
    out["side_firm"] = ((out["new_volume"] > 0) &
                        (out["interval_is_day"] == 0)).astype(int)
    return out


# ------------------------------------------------------------------ screen
def screen(df: pd.DataFrame, min_premium: float | None = None,
           max_dte: int | None = None, atm_band: float | None = None,
           max_oi: int | None = None, min_vol_oi: float | None = None,
           use_interval: bool = True) -> pd.DataFrame:
    """The four filters, as hard gates, plus a magnitude score.

    Gates and score are kept separate on purpose. A gate is a claim that a
    contract does not belong in the conversation at all; a score only orders
    the ones that do. Blending them produces a number where a huge premium can
    buy its way past being three weeks out, which is not what was asked for.

    Every rejected reason is recorded rather than filtered silently, so the
    probe can print how many contracts died at each gate. If one gate is
    killing everything, that shows up as a number instead of an empty page.
    """
    min_premium = config.FLOW_MIN_PREMIUM if min_premium is None else min_premium
    max_dte = config.FLOW_MAX_DTE if max_dte is None else max_dte
    atm_band = config.FLOW_ATM_BAND if atm_band is None else atm_band
    max_oi = config.FLOW_MAX_OI if max_oi is None else max_oi
    min_vol_oi = config.FLOW_MIN_VOL_OI if min_vol_oi is None else min_vol_oi

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "new_premium" not in out.columns or not use_interval:
        out["new_volume"] = out["volume"]
        out["new_premium"] = out["day_premium"]
        for col, default in (("burst", np.nan), ("side_firm", 0),
                             ("interval_is_day", 1)):
            if col not in out.columns:
                out[col] = default

    band = math.log(1.0 + atm_band)
    gates = {
        "dte": out["dte"].between(config.FLOW_MIN_DTE, max_dte),
        "atm": out["log_moneyness"].abs() <= band,
        "premium": out["new_premium"] >= min_premium,
        "low_oi": (out["open_interest"] <= max_oi) |
                  (out["vol_oi"] >= min_vol_oi),
        "new_positioning": out["vol_oi"] >= min_vol_oi,
        "quote": (out["bid"] >= config.FLOW_MIN_BID) &
                 (out["spread_pct"] <= config.FLOW_MAX_SPREAD_PCT),
    }
    for name, ok in gates.items():
        out[f"gate_{name}"] = ok.fillna(False).astype(int)

    keep = np.logical_and.reduce([g.fillna(False).to_numpy() for g in
                                  gates.values()])
    out["flagged"] = keep.astype(int)
    out["score"] = _score(out)
    return out


def _score(df: pd.DataFrame) -> pd.Series:
    """How loud a flag is, once it has passed every gate.

    Three terms, all logarithmic, because the difference between $50k and
    $150k of premium matters and the difference between $3m and $3.1m does
    not. Roughly: one point per tripling of premium above the floor, one point
    per doubling of volume against prior open interest, and up to one point for
    concentration when the day had enough snapshots to measure it.
    """
    prem = pd.to_numeric(df["new_premium"], errors="coerce")
    size = np.log10(np.maximum(prem, 1.0) /
                    config.FLOW_MIN_PREMIUM).clip(0, 2.0) * 2.0

    vo = pd.to_numeric(df["vol_oi"], errors="coerce").fillna(0.0)
    fresh = np.log2(np.maximum(vo, 0.25) /
                    config.FLOW_MIN_VOL_OI).clip(0, 4.0) / 2.0

    burst = pd.to_numeric(df.get("burst"), errors="coerce")
    conc = ((burst - config.FLOW_BURST_ALERT) /
            (1.0 - config.FLOW_BURST_ALERT)).clip(0, 1).fillna(0.0)

    return (size + fresh + conc).round(2)


# --------------------------------------------------------------- roll-up
def by_ticker(flags: pd.DataFrame) -> list[dict]:
    """One row per name, from the contracts that passed.

    The tilt is reported as call premium against put premium, and it is
    deliberately not called bullish or bearish. Bought calls and sold puts look
    the same from here, and so do a directional bet and the long leg of a
    hedge. What the tilt says is where the money went, not what it wanted.
    """
    if flags is None or flags.empty:
        return []
    hits = flags[flags["flagged"] == 1]
    if hits.empty:
        return []

    out = []
    for tkr, g in hits.groupby("ticker"):
        calls = g[g["right"] == "C"]
        puts = g[g["right"] == "P"]
        cprem = float(calls["new_premium"].sum())
        pprem = float(puts["new_premium"].sum())
        total = cprem + pprem
        share = cprem / total if total > 0 else 0.5

        contracts = []
        for _, r in g.sort_values("score", ascending=False).head(8).iterrows():
            contracts.append({
                "contract": r["contract"],
                "right": r["right"],
                "strike": _r(r["strike"], 2),
                "expiry": pd.Timestamp(r["expiry"]).strftime("%Y-%m-%d"),
                "dte": int(r["dte"]),
                "premium": _r(r["new_premium"], 0),
                "volume": _r(r["new_volume"], 0),
                "open_interest": _r(r["open_interest"], 0),
                "vol_oi": _r(r["vol_oi"], 2),
                "mid": _r(r["mid"], 2),
                "burst": _r(r.get("burst"), 2),
                "side": _side(r),
                "iv": _r(r.get("iv"), 3),
                "score": _r(r["score"], 2),
                "stale_mid": int(r.get("mid_from_last", 0) or 0),
            })

        out.append({
            "ticker": tkr,
            "spot": _r(g["spot"].iloc[0], 2),
            "n_contracts": int(len(g)),
            "premium": _r(total, 0),
            "call_premium": _r(cprem, 0),
            "put_premium": _r(pprem, 0),
            "call_share": _r(share, 2),
            "tilt": ("calls" if share >= 0.70 else
                     "puts" if share <= 0.30 else "two-sided"),
            "score": _r(float(g["score"].max()) +
                        0.5 * math.log2(max(len(g), 1)), 2),
            "max_vol_oi": _r(g["vol_oi"].max(), 2),
            "nearest_dte": int(g["dte"].min()),
            "looks_multileg": _multileg(g),
            "contracts": contracts,
        })

    out.sort(key=lambda r: -r["score"])
    return out


def _multileg(g: pd.DataFrame) -> int:
    """A crude check for a position that is not a directional bet.

    Matched volume in a call and a put on the same expiry is a straddle or a
    combo, and reading it as a bullish call buy would be wrong. This catches
    only the obvious case, and only within the flagged set; a real spread
    detector needs the tape. It exists so the page can decline to call
    something directional rather than guess.
    """
    for exp, sub in g.groupby("expiry"):
        c = sub[sub["right"] == "C"]["new_volume"].sum()
        p = sub[sub["right"] == "P"]["new_volume"].sum()
        if c > 0 and p > 0 and abs(c - p) / max(c, p) <= 0.15:
            return 1
    return 0


def _side(r) -> str:
    if not int(r.get("side_firm", 0) or 0):
        return "unknown"
    v = r.get("side_proxy")
    if v is None or not np.isfinite(v):
        return "unknown"
    return "at ask" if v >= 0.4 else "at bid" if v <= -0.4 else "mid"


def _r(v, digits=2):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, digits) if digits else int(round(v))


# ------------------------------------------------------------- snapshots
def snapshot_dir(day: str) -> Path:
    p = config.FLOW_SNAPSHOTS / day
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_snapshot(df: pd.DataFrame, stamp: datetime | None = None) -> Path:
    """Store the near-dated part of a chain so the next scan can difference it."""
    stamp = stamp or datetime.now(timezone.utc)
    keep = df[(df["dte"] >= 0) & (df["dte"] <= config.FLOW_MAX_DTE + 5) &
              (df["log_moneyness"].abs() <= STORE_MONEYNESS)]
    path = snapshot_dir(stamp.strftime("%Y-%m-%d")) / f"{stamp:%H%M}.parquet"
    cols = ["ticker", "contract", "expiry", "right", "strike", "bid", "ask",
            "last", "volume", "open_interest", "iv", "spot", "snapshot_at",
            "dte", "mid", "log_moneyness"]
    keep[[c for c in cols if c in keep.columns]].to_parquet(path, index=False)
    return path


def load_snapshots(day: str) -> list[tuple[str, pd.DataFrame]]:
    """Today's snapshots, oldest first, as (HHMM, frame)."""
    folder = config.FLOW_SNAPSHOTS / day
    if not folder.exists():
        return []
    out = []
    for path in sorted(folder.glob("*.parquet")):
        try:
            out.append((path.stem, pd.read_parquet(path)))
        except Exception as exc:  # noqa: BLE001
            log.warning("unreadable snapshot %s: %s", path, exc)
    return out


def prune(days: int = SNAPSHOT_RETENTION_DAYS) -> int:
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    removed = 0
    for folder in sorted(config.FLOW_SNAPSHOTS.glob("*")):
        if not folder.is_dir():
            continue
        try:
            when = pd.Timestamp(folder.name)
        except ValueError:
            continue
        if when < cutoff:
            for f in folder.glob("*"):
                f.unlink()
                removed += 1
            folder.rmdir()
    return removed


# ------------------------------------------------------- OI confirmation
def confirm_open_interest(day: str, next_day_chain: pd.DataFrame) -> dict:
    """Did the next morning's open interest rise by roughly what traded?

    This is the piece a paid feed gives in real time and free data gives a day
    late: if open interest jumps by about the flagged volume, positions were
    opened and held overnight. If it falls, the volume was closing something.
    A day late is still worth having - it is the difference between a flag that
    survived contact with settlement and one that did not, and the grader can
    use it to split the flags into two groups and see whether the split
    matters.
    """
    flags = load_flags(day)
    if not flags or next_day_chain is None or next_day_chain.empty:
        return {}
    after = (next_day_chain[["contract", "open_interest"]]
             .drop_duplicates("contract")
             .set_index("contract")["open_interest"].to_dict())

    confirmed = checked = 0
    for rec in flags.get("tickers", []):
        for c in rec.get("contracts", []):
            oi_before = c.get("open_interest")
            oi_after = after.get(c["contract"])
            if oi_before is None or oi_after is None:
                continue
            checked += 1
            grew = float(oi_after) - float(oi_before)
            c["oi_change"] = round(grew, 0)
            # Half the traded volume showing up as new open interest is a
            # generous bar and a deliberate one: a trade between a customer
            # opening and a market maker hedging moves OI by less than the
            # full size, and the alternative - demanding the whole volume -
            # rejects genuine opens.
            c["oi_confirmed"] = int(grew >= 0.5 * float(c.get("volume") or 0))
            confirmed += c["oi_confirmed"]
    save_flags(day, flags)
    return {"checked": checked, "confirmed": confirmed,
            "share": round(confirmed / checked, 3) if checked else None}


# ------------------------------------------------------------------ flags
def flags_path(day: str) -> Path:
    return config.FLOW_FLAGS / f"{day}.json"


def save_flags(day: str, payload: dict) -> Path:
    path = flags_path(day)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_flags(day: str) -> dict:
    path = flags_path(day)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        log.warning("unreadable flags for %s: %s", day, exc)
        return {}


def merge_flags(existing: dict, fresh: dict) -> dict:
    """Fold a new scan into the day's record, keeping the loudest per name.

    A name flagged at 10am and again at 2pm is one name with two pieces of
    evidence, not two flags. Keeping the higher-scoring scan and recording both
    times means the day's file stays one row per ticker, which is what the
    grader needs, without losing that it fired twice.
    """
    if not existing:
        return fresh
    by_ticker_ = {r["ticker"]: r for r in existing.get("tickers", [])}
    for rec in fresh.get("tickers", []):
        prev = by_ticker_.get(rec["ticker"])
        # A record written by `scan()` carries no count, which means one. The
        # default has to be 1 rather than 0 or a name flagged in two scans
        # reports as having fired once.
        rec["times_flagged"] = (prev.get("times_flagged", 1) + 1
                                if prev is not None else 1)
        if prev is None or rec["score"] >= prev.get("score", 0):
            rec["first_seen"] = (
                prev.get("first_seen", existing.get("scan_at"))
                if prev is not None else fresh.get("scan_at"))
            by_ticker_[rec["ticker"]] = rec
        else:
            prev["times_flagged"] = rec["times_flagged"]

    out = dict(existing)
    out["tickers"] = sorted(by_ticker_.values(), key=lambda r: -r["score"])
    out["scans"] = sorted(set((existing.get("scans") or []) +
                              (fresh.get("scans") or [])))
    out["scan_at"] = fresh.get("scan_at")
    out["coverage"] = fresh.get("coverage", existing.get("coverage"))
    return out


# -------------------------------------------------------------------- scan
def scan(tickers: list[str], prefer: str = "cboe",
         chain: pd.DataFrame | None = None) -> dict:
    """One pass: fetch, difference against the last snapshot, screen, store."""
    stamp = datetime.now(timezone.utc)
    day = stamp.strftime("%Y-%m-%d")

    errors: dict[str, str] = {}
    if chain is None:
        chain, errors = chains.fetch_many(tickers, prefer=prefer)
    quality = chains.check(chain)

    rich = enrich(chain)
    earlier = load_snapshots(day)
    prev = earlier[-1][1] if earlier else None
    stepped = interval(rich, prev)
    flags = screen(stepped)

    try:
        save_snapshot(rich, stamp)
    except Exception as exc:  # noqa: BLE001 - a failed store is not a failed scan
        log.warning("could not store snapshot: %s", exc)

    records = by_ticker(flags)
    gate_counts = {c.replace("gate_", ""): int(flags[c].sum())
                   for c in flags.columns if c.startswith("gate_")}

    return {
        "asof": day,
        "scan_at": stamp.isoformat(timespec="seconds"),
        "scans": [stamp.strftime("%H%M")],
        "tickers": records,
        "coverage": {
            "requested": len(tickers),
            "with_chain": int(chain["ticker"].nunique()) if len(chain) else 0,
            "failed": len(errors),
            "contracts": int(len(chain)),
            "snapshots_today": len(earlier) + 1,
            "first_scan_of_day": prev is None,
            "chain_quality": quality,
            "gate_survivors": gate_counts,
            "flagged_contracts": int(flags["flagged"].sum()) if len(flags) else 0,
        },
        "errors": dict(list(errors.items())[:20]),
    }
