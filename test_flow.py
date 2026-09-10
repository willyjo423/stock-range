"""Offline checks of the options-flow scan and its grader. No network.

The checks that matter are the negative ones. It is easy to write a screen that
flags the planted block; the useful question is whether it declines to flag the
six decoys that look like it in one respect each. And on the grading side, the
useful question is whether a grader handed data with no effect in it says so -
a measurement that always finds something is the failure mode this project has
hit three times already.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

import chains
import config
import flow
import flow_fixtures as fx
import flow_grade

PASS = FAIL = 0
FAILURES: list[str] = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}   {detail}")


def section(t):
    print(f"\n{t}\n{'-' * 66}")


def _flagged(volume_scale=1.0, prev=None):
    raw = fx.chain(volume_scale=volume_scale)
    rich = flow.enrich(raw, asof=fx.ASOF)
    prev_rich = flow.enrich(prev, asof=fx.ASOF) if prev is not None else None
    return flow.screen(flow.interval(rich, prev_rich))


def names(df):
    return set(df[df["flagged"] == 1]["ticker"])


# ------------------------------------------------------------ OCC parsing
def test_occ():
    section("CONTRACT SYMBOLS")
    m = chains.parse_occ("AAPL240920C00220000")
    check("root, expiry, right and strike come out of the symbol",
          m and m["root"] == "AAPL" and m["right"] == "C"
          and m["strike"] == 220.0
          and m["expiry"] == pd.Timestamp("2024-09-20"), str(m))
    check("a fractional strike survives the round trip",
          chains.parse_occ("F  240920P00012500") is None
          or chains.parse_occ("SPY240920P00012500")["strike"] == 12.5)
    check("a five-digit strike is not truncated",
          chains.parse_occ("NVDA260116C01250000")["strike"] == 1250.0)
    check("junk returns None rather than a wrong answer",
          chains.parse_occ("NOT-A-CONTRACT") is None)

    raw = fx.chain()
    q = chains.check(raw)
    check("the fixture chain agrees with its own symbols",
          q["symbol_disagreements"] == 0, str(q))
    check("and is judged usable", q["usable"], str(q))

    # A chain whose stated strike contradicts its symbol must be caught, since
    # that is what a mis-parsed source looks like.
    bad = raw.copy()
    bad.loc[bad.index[0], "strike"] = 999.0
    check("a strike that contradicts the symbol is reported",
          chains.check(bad)["symbol_disagreements"] == 1)

    blind = raw.copy()
    blind["open_interest"] = np.nan
    check("a source with no open interest is judged unusable",
          not chains.check(blind)["usable"])


# ---------------------------------------------------------------- enrich
def test_enrich():
    section("WHAT ONE SNAPSHOT SUPPORTS")
    rich = flow.enrich(fx.chain(), asof=fx.ASOF)
    block = rich[rich["ticker"] == "BLOCK"].iloc[0]

    check("days to expiry is measured from the as-of date",
          block["dte"] == 7, str(block["dte"]))
    check("an at-the-money strike reads as zero log-moneyness",
          abs(block["log_moneyness"]) < 1e-9)
    check("the mid comes from the quote, not the last trade",
          abs(block["mid"] - 3.00) < 1e-9 and block["mid_from_last"] == 0)
    check("premium is volume x mid x 100",
          abs(block["day_premium"] - 270_000) < 1.0,
          str(block["day_premium"]))
    check("volume against prior open interest is the ratio, not the level",
          abs(block["vol_oi"] - 22.5) < 1e-6, str(block["vol_oi"]))

    # A crossed or missing market must fall back and say that it did.
    broken = fx.chain()
    broken.loc[broken["ticker"] == "BLOCK", ["bid", "ask"]] = [np.nan, np.nan]
    b2 = flow.enrich(broken, asof=fx.ASOF)
    row = b2[b2["ticker"] == "BLOCK"].iloc[0]
    check("a missing quote falls back to the last trade",
          abs(row["mid"] - 3.04) < 1e-9)
    check("and marks the mid as stale rather than hiding it",
          row["mid_from_last"] == 1)

    check("the side proxy reads a print near the offer as at-ask",
          block["side_proxy"] > 0.5, str(block["side_proxy"]))
    put = rich[(rich["ticker"] == "PUTSIDE")].iloc[0]
    check("and a print near the bid as at-bid",
          put["side_proxy"] < -0.5, str(put["side_proxy"]))


# --------------------------------------------------------------- interval
def test_interval():
    section("DIFFERENCING TWO SNAPSHOTS")
    morning, afternoon = fx.two_snapshots()
    m = flow.enrich(morning, asof=fx.ASOF)
    a = flow.enrich(afternoon, asof=fx.ASOF)

    first = flow.interval(m, None)
    check("the first scan of the day treats the interval as the whole day",
          bool((first["interval_is_day"] == 1).all()))
    check("and refuses to report a burst it never measured",
          bool(first["burst"].isna().all()))
    check("the side proxy is not called firm on the first scan",
          int(first["side_firm"].sum()) == 0)

    stepped = flow.interval(a, m)
    block = stepped[stepped["ticker"] == "BLOCK"].iloc[0]
    drib = stepped[stepped["ticker"] == "DRIBBLE"].iloc[0]
    check("a block that traded entirely in one interval bursts at 1.0",
          abs(block["burst"] - 1.0) < 1e-9, str(block["burst"]))
    check("an even dribble bursts at about a half",
          abs(drib["burst"] - 0.5) < 0.02, str(drib["burst"]))
    check("this is the only free way to tell a block from a dribble",
          block["burst"] > drib["burst"] + 0.4)
    check("new volume is the difference, not the total",
          abs(block["new_volume"] - 900) < 1e-6)

    # A source that revises a figure downward must not produce a negative.
    revised = a.copy()
    revised.loc[revised["ticker"] == "BLOCK", "volume"] = 10.0
    back = flow.interval(revised, flow.enrich(fx.chain(), asof=fx.ASOF))
    check("a downward revision clips to zero rather than going negative",
          bool((back["new_volume"] >= 0).all()))


# ----------------------------------------------------------------- screen
def test_gates():
    section("THE FOUR FILTERS, AND THE SIX DECOYS")
    got = names(_flagged())

    check("the planted block is flagged", "BLOCK" in got, str(got))
    check("a month-out expiry is rejected on days to expiry",
          "FARDATE" not in got)
    check("a 25% out-of-the-money strike is rejected as not at the money",
          "OTM" not in got)
    check("a tenth of the premium is rejected on size", "SMALL" not in got)
    check("the same premium into 50,000 open interest is rejected",
          "DRIBBLE" not in got)
    check("an uninvestable 0.02/2.00 market is rejected on quote quality",
          "WIDE" not in got)
    check("the hundred contracts that never traded are all rejected",
          "QUIET" not in got)

    df = _flagged()
    q = df[df["ticker"] == "QUIET"]
    check("and rejected for the right reason - no premium",
          int(q["gate_premium"].sum()) == 0)

    d = df[df["ticker"] == "DRIBBLE"].iloc[0]
    check("the crowded contract fails specifically on new positioning",
          d["gate_premium"] == 1 and d["gate_new_positioning"] == 0)

    # Each gate must be doing work. If loosening one changes nothing, it is
    # decoration and the decoy it was meant to catch is getting through by
    # accident.
    loose = flow.screen(flow.interval(flow.enrich(fx.chain(), asof=fx.ASOF),
                                      None), max_dte=45)
    check("loosening the expiry gate admits the month-out decoy",
          "FARDATE" in names(loose))
    loose = flow.screen(flow.interval(flow.enrich(fx.chain(), asof=fx.ASOF),
                                      None), atm_band=0.40)
    check("loosening the moneyness gate admits the far strike",
          "OTM" in names(loose))
    loose = flow.screen(flow.interval(flow.enrich(fx.chain(), asof=fx.ASOF),
                                      None), min_premium=5_000)
    check("loosening the premium gate admits the small trade",
          "SMALL" in names(loose))
    loose = flow.screen(flow.interval(flow.enrich(fx.chain(), asof=fx.ASOF),
                                      None), max_oi=10 ** 9, min_vol_oi=0.0)
    check("loosening the open-interest gate admits the crowded contract",
          "DRIBBLE" in names(loose))


def test_score():
    section("SCORING")
    df = _flagged()
    hits = df[df["flagged"] == 1].set_index("ticker")
    check("a bigger position outscores a smaller one",
          hits.loc["BLOCK", "score"] > hits.loc["PUTSIDE", "score"],
          f'{hits["score"].to_dict()}')

    morning, afternoon = fx.two_snapshots()
    with_burst = flow.screen(flow.interval(
        flow.enrich(afternoon, asof=fx.ASOF),
        flow.enrich(morning, asof=fx.ASOF)))
    no_burst = _flagged()
    b1 = float(with_burst[with_burst["ticker"] == "BLOCK"]["score"].iloc[0])
    b0 = float(no_burst[no_burst["ticker"] == "BLOCK"]["score"].iloc[0])
    check("concentration in one interval raises the score",
          b1 > b0, f"{b1} vs {b0}")
    check("and an unmeasured burst does not quietly count as concentrated",
          b0 < b1)


# ---------------------------------------------------------------- roll-up
def test_rollup():
    section("PER TICKER")
    recs = {r["ticker"]: r for r in flow.by_ticker(_flagged())}
    check("a call-only flag tilts to calls",
          recs["BLOCK"]["tilt"] == "calls", str(recs.get("BLOCK")))
    check("a put-only flag tilts to puts",
          recs["PUTSIDE"]["tilt"] == "puts")
    check("matched call and put volume is not called directional",
          recs["STRADDLE"]["tilt"] == "two-sided"
          and recs["STRADDLE"]["looks_multileg"] == 1,
          str(recs.get("STRADDLE")))
    check("premium adds up across the flagged contracts",
          abs(recs["STRADDLE"]["premium"] - 240_000) < 1.0,
          str(recs["STRADDLE"]["premium"]))
    check("the loudest name sorts first",
          flow.by_ticker(_flagged())[0]["ticker"] == "BLOCK")
    check("each row carries the contracts behind it",
          len(recs["BLOCK"]["contracts"]) == 1)


def test_merge():
    section("FOLDING SCANS TOGETHER")
    a = {"asof": "2026-03-02", "scan_at": "T1", "scans": ["1000"],
         "tickers": [{"ticker": "BLOCK", "score": 3.0},
                     {"ticker": "PUTSIDE", "score": 2.0}]}
    b = {"asof": "2026-03-02", "scan_at": "T2", "scans": ["1400"],
         "tickers": [{"ticker": "BLOCK", "score": 5.0},
                     {"ticker": "NEW", "score": 1.0}]}
    m = flow.merge_flags(a, b)
    by = {r["ticker"]: r for r in m["tickers"]}
    check("a name flagged twice stays one row", len(m["tickers"]) == 3)
    check("and keeps the louder of the two scores", by["BLOCK"]["score"] == 5.0)
    check("but records that it fired more than once",
          by["BLOCK"]["times_flagged"] == 2, str(by["BLOCK"]))
    check("both scan times are kept", m["scans"] == ["1000", "1400"])
    check("a name only seen in the earlier scan survives", "PUTSIDE" in by)


# ---------------------------------------------------------------- grading
def test_grader_finds_a_planted_effect():
    section("GRADER, WITH AN EFFECT PLANTED")
    df = fx.graded_frame(n_days=120, n_names=90, flag_rate=0.08, effect=0.55)
    s = flow_grade.summarise(df)
    check("it has enough flagged observations to speak", s["enough"],
          str(s["n_flagged"]))
    check("flagged names leave the band more often",
          s["breakout"]["gap"] > 0, str(s["breakout"]))
    check("and the difference clears the usual bar",
          s["breakout"]["z"] > 2.0, str(s["breakout"]))
    check("the continuous reading agrees",
          s["move_size"]["t"] > 2.0, str(s["move_size"]))
    check("the report says what was found",
          "behave differently" in flow_grade.report(s))


def test_grader_reports_a_null():
    section("GRADER, WITH NOTHING PLANTED")
    df = fx.graded_frame(n_days=200, n_names=120, flag_rate=0.08, effect=0.0)
    s = flow_grade.summarise(df)
    check("plenty of observations", s["n_flagged"] > 500, str(s["n_flagged"]))
    check("no breakout effect is claimed",
          abs(s["breakout"]["z"]) < 2.0, str(s["breakout"]))
    check("no move-size effect is claimed",
          abs(s["move_size"]["t"]) < 2.0, str(s["move_size"]))
    txt = flow_grade.report(s)
    check("the report says so plainly", "No effect at the usual bar" in txt)
    check("and says what to do about it", "stop running the scan" in txt)


def test_grader_admits_when_it_cannot_tell():
    section("GRADER, WITH TOO LITTLE TO GO ON")
    df = fx.graded_frame(n_days=8, n_names=30, flag_rate=0.04, effect=0.0)
    s = flow_grade.summarise(df)
    check("it refuses to call a handful of rows a result", not s["enough"],
          str(s["n_flagged"]))
    txt = flow_grade.report(s)
    check("and says so instead of printing a verdict",
          "NOT ENOUGH YET" in txt)
    check("with the number it would need", "needs about" in txt)

    p = flow_grade.power(156, effect=0.08)
    check("the power arithmetic is the plain one",
          p["need_for_effect"] == 157 or p["need_for_effect"] == 156,
          str(p))
    check("a small sample reports a large minimum detectable gap",
          flow_grade.power(25)["detectable_now"] > 0.15)
    check("a large one reports a small gap",
          flow_grade.power(2500)["detectable_now"] < 0.03)


def test_grader_is_not_fooled_by_volatility():
    section("THE TRAP THIS GRADER IS BUILT TO AVOID")
    # Flagged names are all high-volatility names. Measured in raw percentage
    # terms they move far more, and any screen selecting them would look
    # brilliant. Measured against each stock's own band - which is what
    # `abs_move_z` is - the effect must vanish.
    rng = np.random.default_rng(3)
    rows = []
    for d in pd.date_range("2026-01-05", periods=120, freq="B"):
        for i in range(80):
            wild = i < 20
            flagged = int(wild and rng.random() < 0.30)
            sigma = 0.08 if wild else 0.02          # daily vol, raw
            raw = rng.normal(0, sigma)
            z = raw / (sigma * 1.5)                 # the stock's own band
            rows.append({"asof": d, "settle_date": d, "ticker": f"T{i}",
                         "flagged": flagged,
                         "flag_score": 3.0 if flagged else np.nan,
                         "tilt": None, "multileg": 0,
                         "oi_confirmed": np.nan,
                         "raw_move": abs(raw), "move_z": z,
                         "abs_move_z": abs(z),
                         "breakout": int(abs(z) > 1.0),
                         "above": int(z > 1.0), "below": int(z < -1.0),
                         "inside": bool(abs(z) <= 1.0)})
    df = pd.DataFrame(rows)
    raw_gap = (df[df.flagged == 1]["raw_move"].mean()
               - df[df.flagged == 0]["raw_move"].mean())
    s = flow_grade.summarise(df)
    check("in raw terms the flagged names move far more",
          raw_gap > 0.02, f"{raw_gap:.4f}")
    check("but against their own bands the effect disappears",
          abs(s["move_size"]["t"]) < 2.5, str(s["move_size"]))
    check("which is the whole reason the band is the yardstick",
          abs(s["breakout"]["z"]) < 2.5, str(s["breakout"]))


def test_grader_needs_both_groups():
    section("CONTROL GROUP")
    df = fx.graded_frame(n_days=60, n_names=40, flag_rate=1.0, effect=0.0)
    s = flow_grade.summarise(df)
    check("with everything flagged there is nothing to compare against",
          not s.get("breakout"), str(s.get("breakout")))
    check("and the grader does not invent a comparison against 50%",
          s["n_control"] == 0)


# ------------------------------------------------------------------- page
def test_page():
    section("THE PAGE")
    import flow_dashboard
    payload = {
        "asof": "2026-03-02",
        "generated_at": "2026-03-02T20:30:00",
        "scans": ["1000", "1400"],
        "tickers": flow.by_ticker(_flagged()),
        "coverage": {"requested": 500, "with_chain": 486, "failed": 14,
                     "contracts": 120_000, "snapshots_today": 2,
                     "flagged_contracts": 9},
    }
    html = flow_dashboard.render(payload)
    check("the flagged names appear", "BLOCK" in html)
    check("premium is formatted for reading", "$270" in html or "270k" in html)
    check("coverage is stated, including the failures", "14" in html)
    check("the page says premium is not a single trade",
          "however many hands" in html or "not one trade" in html)
    check("the page says the tape is not available",
          "tape" in html.lower())
    check("it declines to call a straddle directional",
          "two-sided" in html)
    check("it links to the forward grading", "flow_results.html" in html)
    check("an empty scan still renders",
          "<html" in flow_dashboard.render({"asof": "2026-03-02",
                                            "tickers": [], "coverage": {}}))


def main():
    print("Options flow - offline checks")
    for fn in (test_occ, test_enrich, test_interval, test_gates, test_score,
               test_rollup, test_merge, test_grader_finds_a_planted_effect,
               test_grader_reports_a_null,
               test_grader_admits_when_it_cannot_tell,
               test_grader_is_not_fooled_by_volatility,
               test_grader_needs_both_groups, test_page):
        try:
            fn()
        except Exception:
            global FAIL
            FAIL += 1
            FAILURES.append(f"{fn.__name__} raised")
            print(f"  CRASH in {fn.__name__}")
            traceback.print_exc()
    print(f"\n{'=' * 66}\n{PASS} passed, {FAIL} failed")
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
