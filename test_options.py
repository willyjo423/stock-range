"""Offline checks of the implied-move comparison and the options page.

The unit conversion is the thing worth testing hardest. A middle-half range and
a one-sigma move differ by a factor of about 1.48, and getting that wrong does
not crash anything - it quietly labels every stock on the page "expensive",
which looks like a finding rather than a bug.
"""
from __future__ import annotations

import math
import sys
import traceback

import numpy as np
import pandas as pd

import flow
import flow_fixtures as fx
import implied
import options_dashboard

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


# ------------------------------------------------------------- unit maths
def test_model_sigma():
    section("TURNING A MIDDLE-HALF RANGE INTO ONE SIGMA")
    # Build a band from a known sigma and check it comes back.
    spot, sigma = 100.0, 0.05
    low = spot * math.exp(-implied.QUARTILE_Z * sigma)
    high = spot * math.exp(implied.QUARTILE_Z * sigma)
    got = implied.model_sigma_pct(low, high)
    check("a band built from 5% sigma reads back as 5%",
          abs(got - 5.0) < 1e-6, f"{got}")

    # The real AAPL numbers from the page.
    got = implied.model_sigma_pct(323.67, 339.83)
    check("AAPL's published week converts to about 3.6%",
          abs(got - 3.61) < 0.05, f"{got}")

    check("the conversion widens the band, never narrows it",
          implied.model_sigma_pct(90, 110) > 100 * (math.log(110 / 90) / 2))
    check("a backwards band is refused",
          implied.model_sigma_pct(110, 90) is None)
    check("a zero price is refused", implied.model_sigma_pct(0, 10) is None)

    # The failure this test exists for: forgetting the conversion makes the
    # middle-half look about a third smaller than one sigma, so implied always
    # wins and every row reads "expensive".
    raw = 100 * (math.log(339.83 / 323.67) / 2)
    check("skipping the conversion would understate the move by ~32%",
          abs(raw / implied.model_sigma_pct(323.67, 339.83) - 0.6745) < 0.01,
          f"{raw}")


def _chain(spot=100.0, iv=0.40, dte=7, strikes=(95, 100, 105), price=2.0):
    rows = []
    for k in strikes:
        for right in ("C", "P"):
            rows.append({
                "ticker": "TEST", "contract": f"TEST{k}{right}",
                "expiry": pd.Timestamp("2026-03-09"), "right": right,
                "strike": float(k), "bid": price - 0.05, "ask": price + 0.05,
                "last": price, "volume": 10.0, "open_interest": 100.0,
                "iv": iv, "spot": spot,
                "snapshot_at": pd.Timestamp("2026-03-02T15:30:00Z"),
                "source": "fixture",
            })
    df = pd.DataFrame(rows)
    df["mid"] = price
    df["spread_pct"] = 0.10 / price
    df["log_moneyness"] = np.log(df["strike"] / spot)
    df["dte"] = dte
    return df


def test_implied():
    section("READING AN IMPLIED MOVE OFF A CHAIN")
    ch = _chain(iv=0.40, dte=7)
    got = implied.implied_sigma_pct(ch, 100.0, 5, dte=7)
    want = 100 * 0.40 * math.sqrt(5 / 252)
    check("implied volatility is scaled to the model's horizon",
          abs(got["implied_move_pct"] - want) < 0.01,
          f"{got.get('implied_move_pct')} vs {want:.3f}")

    # The assertion that was here compared 5/252 against 7/365 and expected
    # them to differ. They barely do - 5 trading days IS about 7 calendar days,
    # so both conventions agree when applied consistently. The real trap is
    # mixing them: using the expiry's calendar days against a 252-day year
    # overstates a one-week move by about 18%. So the property worth asserting
    # is that the HORIZON drives the scaling and the expiry only chooses which
    # volatility to read.
    far_expiry = implied.implied_sigma_pct(_chain(iv=0.40, dte=30), 100.0, 5,
                                           dte=30)
    check("the horizon sets the scaling, not the expiry it was read from",
          abs(far_expiry["implied_move_pct"] - got["implied_move_pct"]) < 0.01,
          f'{far_expiry.get("implied_move_pct")} vs {got["implied_move_pct"]}')
    mixed = 100 * 0.40 * math.sqrt(7 / 252)
    check("mixing calendar days into a trading year would overstate by ~18%",
          abs(mixed / got["implied_move_pct"] - 1.18) < 0.02, f"{mixed}")

    month = implied.implied_sigma_pct(_chain(iv=0.40, dte=30), 100.0, 21, dte=30)
    check("a longer horizon implies a larger move",
          month["implied_move_pct"] > got["implied_move_pct"])
    check("and it scales with the square root of time, not linearly",
          abs(month["implied_move_pct"] / got["implied_move_pct"]
              - math.sqrt(21 / 5)) < 0.01)

    # Quality gates: unusable quotes must produce nothing rather than a number.
    wide = _chain()
    wide["spread_pct"] = 0.9
    check("an uninvestable spread yields no implied move",
          implied.implied_sigma_pct(wide, 100.0, 5, 7) == {})
    nobid = _chain()
    nobid["bid"] = 0.0
    check("a zero bid yields no implied move",
          implied.implied_sigma_pct(nobid, 100.0, 5, 7) == {})
    silly = _chain(iv=9.0)
    check("an impossible implied volatility is rejected, not used",
          implied.implied_sigma_pct(silly, 100.0, 5, 7) == {})
    calls_only = _chain()
    calls_only = calls_only[calls_only["right"] == "C"]
    check("one-sided chains are refused - both sides are needed",
          implied.implied_sigma_pct(calls_only, 100.0, 5, 7) == {})


def test_expiry_choice():
    section("PICKING AN EXPIRY")
    near, far = _chain(dte=6), _chain(dte=45)
    both = pd.concat([near, far], ignore_index=True)
    _, dte = implied.pick_expiry(both, 5)
    check("a one-week horizon takes the near expiry", dte == 6, str(dte))
    _, dte = implied.pick_expiry(both, 21)
    check("a one-month horizon takes the further one", dte == 45, str(dte))
    _, dte = implied.pick_expiry(pd.DataFrame(), 5)
    check("an empty chain returns nothing rather than raising", dte is None)


def test_compare():
    section("THE COMPARISON")
    c = implied.compare(4.0, {"implied_move_pct": 6.0})
    check("a 50% richer option reads as expensive", c["verdict"] == "expensive")
    check("and the gap is stated in percent", c["gap_pct"] == 50, str(c))
    check("the ratio is implied over model", abs(c["ratio"] - 1.5) < 1e-9)

    check("a cheaper option reads as cheap",
          implied.compare(6.0, {"implied_move_pct": 4.0})["verdict"] == "cheap")
    check("a close match reads as fair",
          implied.compare(5.0, {"implied_move_pct": 5.1})["verdict"] == "fair")
    check("no implied move yields no comparison",
          implied.compare(5.0, {}) == {})
    check("no model range yields no comparison",
          implied.compare(None, {"implied_move_pct": 5.0}) == {})

    # A stock whose model band and implied move genuinely agree must not be
    # called anything. This is the test that would fail if the unit conversion
    # were dropped.
    spot, sigma = 100.0, 0.06
    band = implied.model_sigma_pct(spot * math.exp(-implied.QUARTILE_Z * sigma),
                                   spot * math.exp(implied.QUARTILE_Z * sigma))
    iv_needed = sigma / math.sqrt(5 / 252)
    imp = implied.implied_sigma_pct(_chain(iv=iv_needed), spot, 5, 7)
    agree = implied.compare(band, imp)
    check("a market priced exactly at the model reads as fair",
          agree["verdict"] == "fair", str(agree))


def test_lean():
    section("THE LEAN")
    up = implied.lean({"rel_tilt_pct": 0.8, "skew": 0.2}, None)
    check("two inputs pointing up give an up lean", up["direction"] == "up")
    check("and two agreeing reads as moderate", up["strength"] == "moderate",
          str(up))

    one = implied.lean({"rel_tilt_pct": 0.8, "skew": 0.0}, None)
    check("a single input reads as weak", one["strength"] == "weak", str(one))

    split = implied.lean({"rel_tilt_pct": 0.8, "skew": -0.2}, None)
    check("inputs pointing opposite ways give no lean",
          split["direction"] == "flat")
    check("and the page says why", split["strength"] == "signals disagree")

    quiet = implied.lean({"rel_tilt_pct": 0.05, "skew": 0.01}, None)
    check("nothing meaningful gives no lean", quiet["direction"] == "flat")
    check("and says there is nothing to say",
          quiet["strength"] == "nothing to say", str(quiet))

    # Flow: a straddle must never be read as directional.
    straddle = implied.lean(
        {"rel_tilt_pct": 0.0, "skew": 0.0},
        {"call_share": 0.52, "call_premium": 2.8e6, "put_premium": 2.6e6,
         "looks_multileg": 1})
    drv = [d for d in straddle["drivers"] if d["name"] == "options flow"][0]
    check("matched legs are not called a direction", drv["direction"] == "flat")
    check("and the page says it looks like a straddle",
          "straddle" in drv["detail"])

    calls = implied.lean({"rel_tilt_pct": 0.0, "skew": 0.0},
                         {"call_share": 0.85, "call_premium": 4.2e6,
                          "put_premium": 0.7e6, "looks_multileg": 0})
    drv = [d for d in calls["drivers"] if d["name"] == "options flow"][0]
    check("one-sided call premium votes up", drv["direction"] == "up")
    check("flow is marked as ungraded", drv["evidence"] == "ungraded")
    tilt = [d for d in calls["drivers"] if d["name"] == "model tilt"][0]
    check("the model tilt is marked as measured", tilt["evidence"] == "measured")


def test_page():
    section("THE PAGE")
    payload = {
        "asof": "2026-09-14", "generated_at": "2026-09-14T22:30:00",
        "coverage": {"forecast": 503, "priced": 486, "no_chain": 17},
        "tickers": [{
            "ticker": "AAPL", "close": 332.27,
            "horizons": {
                "1 week": {"days": 5, "low": 323.67, "high": 339.83,
                           "model_move_pct": 3.61, "implied_move_pct": 5.10,
                           "ratio": 1.413, "gap_pct": 41.0,
                           "verdict": "expensive", "earnings_inside": 0},
                "1 month": {"days": 21, "low": 316.12, "high": 350.40,
                            "model_move_pct": 7.7, "implied_move_pct": 7.8,
                            "ratio": 1.01, "gap_pct": 1.0, "verdict": "fair"},
            },
            "lean": implied.lean({"rel_tilt_pct": 0.6, "skew": 0.0},
                                 {"call_share": 0.82, "call_premium": 4.2e6,
                                  "put_premium": 0.9e6, "looks_multileg": 0}),
        }],
    }
    h = options_dashboard.render(payload)
    check("the ticker and price appear", "AAPL" in h and "332.27" in h)
    check("the verdict is stated in words", "Options look expensive" in h)
    check("the gap is explained as a sentence", "41% more" in h)
    check("both moves are shown", "3.6%" in h and "5.1%" in h)
    check("the range is still there in dollars", "323.67" in h)
    check("the lean appears once, not per period", h.count("Leans up") == 1)
    check("the measured input is labelled", "MEASURED" in h)
    check("the ungraded input is labelled", "NOT YET GRADED" in h)
    check("the unit conversion is explained on the page",
          "middle half" in h and "one-standard-deviation" in h)
    check("the page says a gap is not an answer", "not an answer" in h)
    check("coverage is stated including the failures", "17" in h)

    filtered = options_dashboard.render(payload, min_gap=0.9)
    check("a high gap filter can empty the page", "Nothing to show" in filtered)
    check("an empty payload still renders",
          "<html" in options_dashboard.render({"asof": "x", "tickers": []}))


def test_no_quotes_is_visible():
    section("WHEN THERE ARE NO QUOTES")
    payload = {"asof": "2026-09-14", "coverage": {},
               "tickers": [{"ticker": "XYZ", "close": 10.0,
                            "horizons": {"1 week": {
                                "days": 5, "low": 9.5, "high": 10.5,
                                "model_move_pct": 7.4}},
                            "lean": {}}]}
    h = options_dashboard.render(payload)
    check("a name with no chain still shows its range", "9.50" in h)
    check("and says plainly that nothing was comparable",
          "nothing to" in h.lower())
    check("it is not labelled expensive or cheap",
          "Options look" not in h)


def main():
    print("Options view - offline checks")
    for fn in (test_model_sigma, test_implied, test_expiry_choice,
               test_compare, test_lean, test_page, test_no_quotes_is_visible):
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
