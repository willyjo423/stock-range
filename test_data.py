"""Offline verification of the data layer. No network.

The checks that matter most are the leakage ones and the audit one. A
volatility feature that peeks one day into the future looks superb and is
worthless, and an unadjusted split inflates every number quietly. Both are
silent failures, so both get an explicit test.

    python test_data.py
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

import config
import fixtures
import prices
import universe

PASS = FAIL = 0
FAILURES: list[str] = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}   {detail}")


def section(t):
    print(f"\n{t}\n{'-' * 68}")


# ------------------------------------------------------------------ tickers
def test_tickers() -> None:
    section("TICKER NORMALISATION")
    check("class shares get a dash", universe.normalise_ticker("BRK.B") == "BRK-B",
          "BRK.B and BF.B are two of the larger names; the dot silently "
          "drops them")
    check("whitespace and case handled",
          universe.normalise_ticker("  aapl ") == "AAPL")
    check("ordinary tickers untouched", universe.normalise_ticker("MSFT") == "MSFT")


def test_universe() -> None:
    section("UNIVERSE AND SURVIVORSHIP")
    hist = fixtures.make_universe_history()
    uni = universe.Universe(hist, list(fixtures.TICKERS[:30]))

    check("reports itself survivorship-safe", uni.survivorship_safe)
    early = uni.members_on("2014-06-01")
    late = uni.members_on("2023-06-01")
    check("membership is a function of date", early != late,
          "if these match, the history is not being used")
    check("membership stays a sensible size",
          20 <= len(early) <= 40 and 20 <= len(late) <= 40,
          f"{len(early)} then {len(late)}")

    ever = uni.all_ever()
    check("more names ever than today", len(ever) > len(uni.current),
          f"{len(ever)} ever vs {len(uni.current)} now - the gap IS the bias")
    check("today's members are all included",
          set(uni.current) <= set(ever))

    blind = universe.Universe(None, list(fixtures.TICKERS[:30]))
    check("a missing history is admitted, not hidden",
          not blind.survivorship_safe)
    check("and it still returns a usable list",
          blind.members_on("2015-01-01") == blind.current)

    s = uni.summary()
    check("summary reports the turnover", s["turnover"] == len(ever) - len(uni.current))


# ------------------------------------------------------------------ returns
def test_returns() -> None:
    section("RETURNS")
    px = fixtures.make_prices(with_splits=False)
    df = prices.add_returns(px)

    check("a return per row bar the first per ticker",
          int(df["ret"].isna().sum()) == df["ticker"].nunique(),
          f"{int(df['ret'].isna().sum())} NaN vs {df['ticker'].nunique()} tickers")
    check("no return crosses tickers",
          df.groupby("ticker")["ret"].apply(lambda s: s.iloc[0]).isna().all(),
          "the first row of each ticker must be NaN, not the previous "
          "company's close")

    sd = float(df["ret"].std())
    check("daily spread is plausible", 0.005 < sd < 0.06, f"{sd:.4f}")
    check("range volatility computed", df["range_vol"].notna().mean() > 0.9)
    check("range volatility is positive", (df["range_vol"].dropna() >= 0).all())


def test_audit() -> None:
    section("SPLIT ADJUSTMENT AUDIT")
    clean = prices.audit(prices.add_returns(fixtures.make_prices(with_splits=False)))
    check("clean data reports almost no impossible days",
          clean["implausible_share"] < 0.001,
          f"{clean['implausible_share']:.5f}")

    dirty = prices.audit(prices.add_returns(fixtures.make_prices(with_splits=True)))
    check("an unadjusted split is detected",
          dirty["split_shaped_moves"] > clean["split_shaped_moves"],
          f"{dirty['split_shaped_moves']} vs {clean['split_shaped_moves']} - "
          f"if these match, the audit cannot see the thing it exists for")
    check("and shows up as an impossible move",
          dirty["implausible_moves"] > clean["implausible_moves"])


# ------------------------------------------------------------------ vol
def test_realized_vol() -> None:
    section("REALIZED VOLATILITY")
    df = prices.add_returns(fixtures.make_prices(with_splits=False))
    rv = prices.realized_vol(df)

    for w in config.RV_WINDOWS:
        check(f"rv_{w} present", f"rv_{w}" in rv.columns)
    vals = rv["rv_21"].dropna()
    check("annualised into a believable range",
          0.05 < vals.median() < 1.2, f"median {vals.median():.3f}")
    check("never negative", (vals >= 0).all())
    check("term-structure slope computed", rv["rv_slope"].notna().mean() > 0.5)

    # The leakage test. A trailing window must not move when the future does.
    cut = rv["date"].quantile(0.7)
    truncated = prices.realized_vol(df[df["date"] <= cut])
    a = rv[rv["date"] <= cut].set_index(["ticker", "date"])["rv_21"]
    b = truncated.set_index(["ticker", "date"])["rv_21"]
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    check("deleting the future changes nothing",
          len(joined) > 1000 and np.allclose(joined.iloc[:, 0], joined.iloc[:, 1]),
          f"compared {len(joined):,} rows - a mismatch means the window is "
          f"seeing forward")

    # A half-empty window must not masquerade as a measurement.
    short = df.groupby("ticker").head(4)
    check("too little history yields no volatility",
          prices.realized_vol(short)["rv_63"].isna().all())


def test_forward_return() -> None:
    section("THE TARGET")
    px = fixtures.make_prices(with_splits=False)
    df = prices.add_returns(px).sort_values(["ticker", "date"]).reset_index(drop=True)
    fwd = prices.forward_return(df, 21)

    check("the last rows of each ticker have no target",
          fwd.isna().sum() >= 21 * df["ticker"].nunique())

    # Spot-check one value against the raw prices it should come from.
    one = df[df["ticker"] == df["ticker"].iloc[0]].reset_index(drop=True)
    f1 = prices.forward_return(one, 21).reset_index(drop=True)
    want = np.log(one["close"].iloc[21] / one["close"].iloc[0])
    check("it is the log return to the horizon close",
          abs(float(f1.iloc[0]) - float(want)) < 1e-12,
          f"{f1.iloc[0]:.8f} vs {want:.8f}")

    check("no target crosses tickers",
          prices.forward_return(df, 5).groupby(df["ticker"]).apply(
              lambda s: s.tail(5).isna().all()).all(),
          "the last days of one ticker must not borrow the next one's prices")


def test_premise() -> None:
    """The relationship the whole project depends on, planted and recovered."""
    section("THE PREMISE, ON PLANTED DATA")
    df = prices.add_returns(fixtures.make_prices(days=3000, with_splits=False))
    rv = prices.realized_vol(df)

    fwd = (rv.sort_values(["ticker", "date"]).groupby("ticker")["ret"]
             .rolling(21, min_periods=10).std()
             .reset_index(level=0, drop=True).shift(-21) * np.sqrt(252))
    rv["fwd_vol"] = fwd
    ok = rv[["rv_21", "fwd_vol"]].dropna()
    corr = float(np.corrcoef(ok["rv_21"], ok["fwd_vol"])[0, 1])
    check("volatility persistence is recovered", corr > 0.35,
          f"corr={corr:+.3f} on {len(ok):,} rows - the fixtures generate "
          f"clustered volatility, so failing here is a code fault")

    # And prove the test has teeth: shuffled returns must destroy it.
    shuffled = df.copy()
    rng = np.random.default_rng(7)
    shuffled["ret"] = rng.permutation(shuffled["ret"].to_numpy())
    srv = prices.realized_vol(shuffled)
    sfwd = (srv.sort_values(["ticker", "date"]).groupby("ticker")["ret"]
              .rolling(21, min_periods=10).std()
              .reset_index(level=0, drop=True).shift(-21) * np.sqrt(252))
    srv["fwd_vol"] = sfwd
    sok = srv[["rv_21", "fwd_vol"]].dropna()
    scorr = float(np.corrcoef(sok["rv_21"], sok["fwd_vol"])[0, 1])
    check("shuffling the returns destroys it", abs(scorr) < 0.15,
          f"corr={scorr:+.3f} - if persistence survives a shuffle, the "
          f"measurement is picking up something structural, not the signal")


def main() -> int:
    print("Range forecast - offline data layer checks")
    for fn in (test_tickers, test_universe, test_returns, test_audit,
               test_realized_vol, test_forward_return, test_premise):
        try:
            fn()
        except Exception:  # noqa: BLE001
            global FAIL
            FAIL += 1
            FAILURES.append(f"{fn.__name__} raised")
            print(f"  CRASH in {fn.__name__}")
            traceback.print_exc()

    print(f"\n{'=' * 68}\n{PASS} passed, {FAIL} failed")
    if FAILURES:
        print("\nfailures:")
        for f in FAILURES:
            print(f"  - {f}")
    print("=" * 68)
    print("\nThese prove the code is self-consistent against a synthetic")
    print("market. Whether the real data is adjusted, how far back it goes,")
    print("and whether volatility actually persists in it are questions only")
    print("`python selftest.py` can answer.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
