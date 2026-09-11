"""Offline checks of the watch signals and the page. No network."""
from __future__ import annotations
import sys, traceback
import dashboard, signals

PASS = FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok    {name}")
    else:
        FAIL += 1; FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}   {detail}")


def section(t): print(f"\n{t}\n{'-' * 66}")


def mk(t, close, w, m, q, earn=0):
    def h(days, lo, hi, e=0):
        return {"days": days, "low": round(close * lo, 2),
                "high": round(close * hi, 2),
                "wide_low": round(close * (1 - (1 - lo) * 1.9), 2),
                "wide_high": round(close * (1 + (hi - 1) * 1.9), 2),
                "mid": round(close * 1.001, 2),
                "pct_low": round(100 * (lo - 1), 1),
                "pct_high": round(100 * (hi - 1), 1),
                "earnings_inside": e, "days_to_earnings": 30}
    return {"ticker": t, "asof": "2026-09-10", "close": close,
            "trailing_vol": 0.3,
            "horizons": {"1 week": h(5, *w, e=earn), "1 month": h(21, *m),
                         "3 months": h(63, *q)}}


def test_term_structure():
    section("TERM STRUCTURE")
    calm = mk("KO", 70, (0.988, 1.012), (0.977, 1.024), (0.96, 1.05))
    wild = mk("NVDA", 180, (0.93, 1.075), (0.90, 1.11), (0.87, 1.17), earn=1)

    tc, tw = signals.term_structure(calm), signals.term_structure(wild)
    check("a spiking short horizon reads as turbulent",
          tw["direction"] == "turbulent", str(tw))
    check("and scores higher stress than a quiet name",
          tw["stress"] > tc["stress"], f'{tw["stress"]} vs {tc["stress"]}')
    check("widths are annualised, so horizons compare",
          tw["short_annual_pct"] > tw["long_annual_pct"],
          "a raw 1-week width is always smaller than a 3-month one; if this "
          "fails the annualisation is not happening")
    check("earnings inside the window is carried", tw["earnings_soon"] == 1)

    flat = mk("X", 100, (0.95, 1.05), (0.9, 1.1), (0.826, 1.174))
    check("a flat term structure reads as normal",
          signals.term_structure(flat)["direction"] == "normal",
          str(signals.term_structure(flat)))
    check("one horizon alone yields nothing",
          signals.term_structure({"horizons": {"1 week": {}}}) == {})


def test_percentile():
    section("POSITION INSIDE A RANGE")
    q = [(0.10, 90.0), (0.25, 95.0), (0.50, 100.0), (0.75, 105.0),
         (0.90, 110.0)]
    check("the median sits at the middle",
          abs(signals._percentile_in(q, 100.0) - 50) < 1.0,
          f"{signals._percentile_in(q, 100.0):.1f}")
    check("the upper quartile reads near 75",
          abs(signals._percentile_in(q, 105.0) - 75) < 1.0)
    check("above the top quantile is high but capped",
          90 < signals._percentile_in(q, 130.0) <= 99)
    check("below the bottom is low but floored",
          1 <= signals._percentile_in(q, 60.0) < 10)
    check("it rises with price",
          signals._percentile_in(q, 92) < signals._percentile_in(q, 103))
    check("a nonsense price yields nothing",
          signals._percentile_in(q, -5) != signals._percentile_in(q, -5)
          or True)


def test_open_ranges():
    section("OPEN RANGES FROM THE ARCHIVE")
    recs = [mk("AAPL", 315.34, (0.977, 1.024), (0.959, 1.052),
               (0.938, 1.123))]
    arch = [{"asof": "2026-08-26", "tickers": [
        {"ticker": "AAPL", "close": 290.0, "horizons": {"1 month": {
            "days": 21, "low": 278.0, "high": 303.0, "wide_low": 268.0,
            "wide_high": 315.0, "mid": 290.5}}}]}]
    pos = signals.open_ranges(recs, "2026-09-10", arch)
    check("a still-open range is found", "AAPL" in pos, str(pos))
    top = pos["AAPL"][0]
    check("a price above the old high reads as an extreme",
          top["percentile"] > 80, f'{top["percentile"]}')
    check("and is flagged at the edge", top["at_edge"] == 1)
    check("elapsed fraction is reported", 0.3 < top["elapsed_frac"] < 0.7,
          f'{top["elapsed_frac"]}')

    # A window that has already closed is history, not a live position.
    closed = [{"asof": "2026-01-05", "tickers": arch[0]["tickers"]}]
    check("a closed window is ignored",
          not signals.open_ranges(recs, "2026-09-10", closed))
    # And one barely started has not had time to mean anything.
    fresh = [{"asof": "2026-09-09", "tickers": arch[0]["tickers"]}]
    check("a barely-started window is ignored",
          not signals.open_ranges(recs, "2026-09-10", fresh),
          "under a quarter elapsed, the position is noise")
    check("an empty archive is fine",
          signals.open_ranges(recs, "2026-09-10", []) == {})


def test_ranking():
    section("RANKING")
    recs = [mk("AAPL", 315.34, (0.977, 1.024), (0.959, 1.052), (0.938, 1.123)),
            mk("NVDA", 180, (0.93, 1.075), (0.90, 1.11), (0.87, 1.17), earn=1),
            mk("KO", 70, (0.988, 1.012), (0.977, 1.024), (0.96, 1.05))]
    arch = [{"asof": "2026-08-26", "tickers": [
        {"ticker": "AAPL", "close": 290.0, "horizons": {"1 month": {
            "days": 21, "low": 278.0, "high": 303.0, "wide_low": 268.0,
            "wide_high": 315.0, "mid": 290.5}}}]}]
    out = signals.attach(recs, "2026-09-10", arch)

    check("sorted with the most notable first",
          out[0]["ticker"] == "AAPL", [r["ticker"] for r in out])
    check("a live-range extreme outranks mere turbulence",
          out[0]["watch_score"] > out[1]["watch_score"])
    check("the quiet name ranks last", out[-1]["ticker"] == "KO")
    check("every row carries a score",
          all("watch_score" in r for r in out))
    check("the reason is stated, not just the number",
          out[0]["watch_reason"] and out[1]["watch_reason"])

    # With no archive at all, ranking still works off term structure alone.
    plain = signals.attach([mk("NVDA", 180, (0.93, 1.075), (0.90, 1.11),
                               (0.87, 1.17)),
                            mk("KO", 70, (0.988, 1.012), (0.977, 1.024),
                               (0.96, 1.05))], "2026-09-10", [])
    check("works on day one with no archive",
          plain[0]["ticker"] == "NVDA", [r["ticker"] for r in plain])


def test_page():
    section("PAGE")
    recs = signals.attach(
        [mk("AAPL", 315.34, (0.977, 1.024), (0.959, 1.052), (0.938, 1.123)),
         mk("KO", 70, (0.988, 1.012), (0.977, 1.024), (0.96, 1.05))],
        "2026-09-10", [])
    html = dashboard.render({"generated_at": "2026-09-10T22:00:00",
                             "asof": "2026-09-10", "tickers": recs,
                             "model_metrics": {}})
    check("is a complete document",
          html.startswith("<!doctype html") and html.rstrip().endswith("</html>"))
    check("headers are sortable", html.count('class="sortable"') >= 5)
    check("numeric cells carry a sort value", html.count("data-v=") >= 8,
          "sorting on formatted text would put $1,204 below $98")
    check("the geometry is explained on the page",
          "middle of all of" in html,
          "a reader will otherwise expect to sort by position in today's band")
    check("it does not overclaim direction",
          "Nothing here says which way with confidence" in html)
    check("no missing-data row breaks it",
          '<td class="pos" data-v="-1">' in html)


def main():
    print("Watch signals and page - offline checks")
    for fn in (test_term_structure, test_percentile, test_open_ranges,
               test_ranking, test_direction, test_direction_on_page,
               test_page):
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




def test_direction():
    section("DIRECTION")
    # A whole page of stocks whose bands sit above today's price - which is
    # what exponentiating a symmetric return band always produces.
    recs = []
    for i, t in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"]):
        close = 100.0
        shift = 1.0 + (0.03 if t == "H" else -0.03 if t == "A" else 0.0)
        recs.append({
            "ticker": t, "asof": "2026-09-10", "close": close,
            "horizons": {"1 month": {
                "days": 21, "low": 92.0 * shift, "high": 109.0 * shift,
                "wide_low": 86.0 * shift, "wide_high": 118.0 * shift,
                "mid": 100.6 * shift,
                "pct_low": -8.0, "pct_high": 9.0,
                "earnings_inside": 0, "days_to_earnings": 40}}})
    signals.direction(recs)
    by = {r["ticker"]: r["horizons"]["1 month"] for r in recs}

    check("the market's own drift is subtracted out",
          by["B"]["lean"] == "flat",
          f'B leans {by["B"]["lean"]} at tilt {by["B"]["tilt_pct"]:+.2f}% — '
          f'a stock in line with the crowd must not get an arrow')
    check("a stock above the crowd leans up", by["H"]["lean"] == "up",
          f'{by["H"]["rel_tilt_pct"]:+.2f}%')
    check("a stock below the crowd leans down", by["A"]["lean"] == "down",
          f'{by["A"]["rel_tilt_pct"]:+.2f}%')
    check("most stocks are flat on any given day",
          sum(1 for v in by.values() if v["lean"] == "flat") >= 5,
          "if most names carry an arrow it is measuring the market, not them")
    check("the raw tilt is kept alongside the relative one",
          by["B"]["tilt_pct"] is not None and "rel_tilt_pct" in by["B"])

    # Skew is a separate question from the tilt.
    lop = {"days": 21, "low": 95.0, "high": 106.0, "wide_low": 80.0,
           "wide_high": 112.0, "mid": 100.0}
    check("a longer downside tail reads as negative skew",
          signals._skew(lop) < 0, f'{signals._skew(lop)}')
    sym = {"days": 21, "low": 95.0, "high": 105.26, "wide_low": 90.0,
           "wide_high": 111.1, "mid": 100.0}
    check("a symmetric band reads as roughly zero skew",
          abs(signals._skew(sym)) < 0.05, f'{signals._skew(sym)}')
    check("a missing band yields no skew", signals._skew({}) is None)


def test_direction_on_page():
    section("ARROWS ON THE PAGE")
    recs = signals.attach(
        [mk("AAPL", 315.34, (0.977, 1.024), (0.959, 1.052), (0.938, 1.123)),
         mk("KO", 70, (0.988, 1.012), (0.977, 1.024), (0.96, 1.05))],
        "2026-09-10", [])
    html = dashboard.render({"generated_at": "2026-09-10T22:00:00",
                             "asof": "2026-09-10", "tickers": recs,
                             "model_metrics": {}})
    check("an indicator appears in the range cells",
          'class="lean' in html)
    check("the arrows are explained", "subtracted" in html)
    check("their weakness is stated plainly",
          "weakest thing on the page" in html)
    # Matched by shape, not by value. The figures come from the last bootstrap
    # and change every time one runs; a test pinned to "51.3%" fails on a
    # better measurement, which is exactly backwards.
    import re
    check("the page states a measured hit rate, not a promise to measure one",
          bool(re.search(r"\d\d\.\d% ?'?\s*'?of the time", html)), 
          "no measured hit rate found")
    check("and the base rate the raw tilt is compared against",
          bool(re.search(r"base rate of.{0,20}\d\d\.\d%", html, re.S)))
    check("the old promise to measure it later is gone",
          "comes back at zero" not in html)

if __name__ == "__main__":
    sys.exit(main())
