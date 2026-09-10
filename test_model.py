"""Offline checks of the feature and model layers. No network.

The fixtures generate volatility that clusters *and reverts*, which is the
behaviour the probe found in real data: a quiet stock does not stay as quiet as
its trailing window suggests. So a naive band should be miscalibrated on this
data in the same direction as on the real thing, and the model should close the
gap. If it does not, that is a code fault, not a market fact.

    python test_model.py
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd
from scipy.stats import norm

import config
import features
import fixtures
import model as M

PASS = FAIL = 0
FAILURES: list[str] = []
_PANEL = {}


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
    print(f"\n{t}\n{'-' * 70}")


# How large a panel the walk-forward tests use.
#
# This is not an arbitrary choice. Measured on these fixtures, the model needs
# a certain amount of *independent* evidence before it beats the naive band at
# all, because estimating conditional quantiles costs variance and the naive
# answer costs none:
#
#     independent windows   pinball vs naive   worst bucket (model / naive)
#              5,614            -2.9%              6.2 / 9.8
#             18,629            +0.2%              2.3 / 9.4
#             51,840            +1.1%              3.2 / 11.0
#
# So a 40-ticker fixture would "prove" the approach does not work, when what it
# actually shows is that 40 tickers is not enough. Conditional calibration
# improves at every size; the pinball crossover needs roughly 15,000 windows.
# The real panel - 500 names over twenty years - has well past 100,000.
#
# (The table above counts every independent window; the walk-forward figure is
# lower because the first years are spent training.)
BIG_TICKERS = [f"T{i:03d}" for i in range(1, 121)]


def panel(horizon: int = 21, days: int = 3200, tickers=None) -> pd.DataFrame:
    tickers = tuple(tickers) if tickers else None
    key = (horizon, days, tickers)
    if key not in _PANEL:
        tk = list(tickers) if tickers else None
        px = fixtures.make_prices(tickers=tk, days=days, with_splits=False,
                                  with_delistings=False)
        base = features.build_panel(px, fixtures.make_market(days=days))
        _PANEL[key] = features.build_for_horizon(
            base, horizon, fixtures.make_earnings(tickers=tk))
    return _PANEL[key]


def big_panel() -> pd.DataFrame:
    return panel(21, 3600, BIG_TICKERS)


# ------------------------------------------------------------------ features
def test_features() -> None:
    section("FEATURES")
    df = panel()

    missing = [c for c in features.ALL_FEATURE_COLUMNS if c not in df.columns]
    check("every declared feature is built", not missing, str(missing))
    check("all features are numeric",
          all(pd.api.types.is_numeric_dtype(df[c])
              for c in features.ALL_FEATURE_COLUMNS))

    # The groups the first real bootstrap showed to be harmful are built and
    # still ablated, but must not reach the shipped model.
    check("market features are excluded from the shipped set",
          not (set(features.MARKET_COLUMNS) & set(features.FEATURE_COLUMNS)),
          "market was the most harmful group at every horizon")
    check("flow features are excluded from the shipped set",
          not (set(features.FLOW_COLUMNS) & set(features.FEATURE_COLUMNS)))
    check("but both are still built, so they can be re-measured",
          set(features.MARKET_COLUMNS + features.FLOW_COLUMNS)
          <= set(features.ALL_FEATURE_COLUMNS))

    filled = df[features.ALL_FEATURE_COLUMNS].notna().mean()
    check("volatility core is populated", filled["log_rv_21"] > 0.85,
          f"{filled['log_rv_21']:.3f}")
    check("market context is populated", filled["log_vix"] > 0.85,
          f"{filled['log_vix']:.3f}")
    check("earnings features are populated", filled["earnings_in_window"] > 0.7,
          f"{filled['earnings_in_window']:.3f}")

    # The target must never appear among the inputs.
    for leak in ("z", "fwd_ret", "scale"):
        check(f"{leak} is not a feature", leak not in features.FEATURE_COLUMNS)

    z = df["z"].dropna()
    check("standardized target is centred near zero", abs(z.median()) < 0.25,
          f"median {z.median():+.3f}")
    check("and has a spread near one", 0.6 < z.std() < 2.5, f"sd {z.std():.3f}")


def test_earnings_window() -> None:
    section("EARNINGS, PER HORIZON")
    px = fixtures.make_prices(days=1500, with_splits=False,
                              with_delistings=False)
    base = features.build_panel(px, fixtures.make_market(days=1500))
    e = fixtures.make_earnings()

    short = features.add_earnings(base, e, 5)["earnings_in_window"].mean()
    long = features.add_earnings(base, e, 63)["earnings_in_window"].mean()
    check("a longer window catches more reports", long > short,
          f"5d {short:.3f} vs 63d {long:.3f}")
    check("a week rarely contains one", short < 0.25, f"{short:.3f}")
    check("a quarter almost always does", long > 0.7, f"{long:.3f}")

    none = features.add_earnings(base, None, 21)
    check("no earnings data yields NaN, not zero",
          none["earnings_in_window"].isna().all(),
          "zero would assert there is no report, which is a different claim")


# -------------------------------------------------------------------- model
def test_quantiles() -> None:
    section("QUANTILE MODEL")
    df = panel()
    X, carry = features.training_matrix(df)
    check("training matrix has rows", len(X) > 20000, f"{len(X)}")

    m = M.RangeModel(horizon=21).fit(X.iloc[:20000], carry["z"].iloc[:20000])
    z = m.predict_z(X.iloc[20000:25000])

    check("one column per quantile", list(z.columns) == list(config.QUANTILES))
    arr = z.to_numpy()
    check("quantiles never cross", bool((np.diff(arr, axis=1) >= 0).all()),
          "a 25th above a 75th makes a range that reads backwards")
    check("the median sits between the quartiles",
          bool(((arr[:, 1] <= arr[:, 2]) & (arr[:, 2] <= arr[:, 3])).all()))
    check("the band is a sane width",
          0.5 < float(np.median(arr[:, 3] - arr[:, 1])) < 3.0,
          f"{np.median(arr[:, 3] - arr[:, 1]):.2f}")

    # Crossing must be actively prevented, not merely absent by luck.
    class Reversed(M.RangeModel):
        def __init__(s):
            super().__init__()
            s.quantiles = [0.25, 0.75]
            s.models = {}
    r = Reversed()
    n = 40

    class _Const:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.full(len(X), self.v)

    r.models = {0.25: _Const(2.0), 0.75: _Const(-2.0)}   # deliberately crossed
    out = r.predict_z(X.iloc[:n])
    check("a crossed pair is repaired, not passed through",
          bool((out[0.75].to_numpy() >= out[0.25].to_numpy()).all()),
          "the sort in predict_z is what guarantees this")

    scale = carry["scale"].iloc[20000:25000]
    close = carry["close"].iloc[20000:25000]
    px = m.predict(X.iloc[20000:25000], scale, close)
    check("price levels are produced", "px_q25" in px.columns)
    check("price quantiles keep their order",
          bool((px["px_q75"] >= px["px_q25"]).all()))
    check("prices are positive", bool((px["px_q10"] > 0).all()))


def test_pinball() -> None:
    section("PINBALL LOSS")
    y = np.array([1.0, 1.0, 1.0, 1.0])
    check("symmetric at the median",
          abs(M.pinball(y, np.zeros(4), 0.5) - M.pinball(y, np.full(4, 2.0), 0.5))
          < 1e-12)
    # Under-predicting a high quantile must cost more than over-predicting it.
    under = M.pinball(y, np.full(4, 0.0), 0.9)
    over = M.pinball(y, np.full(4, 2.0), 0.9)
    check("the 90th punishes under-prediction harder", under > over,
          f"under {under:.3f} vs over {over:.3f}")
    # The true quantile of a sample minimises it.
    rng = np.random.default_rng(3)
    s = rng.normal(size=20000)
    truth = float(np.quantile(s, 0.25))
    losses = {d: M.pinball(s, np.full(len(s), truth + d), 0.25)
              for d in (-0.3, -0.1, 0.0, 0.1, 0.3)}
    check("it is minimised at the true quantile",
          min(losses, key=losses.get) == 0.0, str({k: round(v, 4) for k, v in losses.items()}))


def test_non_overlapping() -> None:
    section("OVERLAP")
    df = panel()
    thin = M.non_overlapping(df.dropna(subset=["z"]), 21)
    full = df.dropna(subset=["z"])
    ratio = len(thin) / len(full)
    check("thinning keeps roughly one row in every horizon",
          0.03 < ratio < 0.07, f"kept {ratio:.3f}")

    gaps = (thin.sort_values(["ticker", "date"])
                .groupby("ticker")["date"].diff().dt.days.dropna())
    check("consecutive kept windows do not overlap", gaps.min() >= 21,
          f"smallest gap {gaps.min()} days - anything under the horizon means "
          f"the windows share days")
    check("every ticker survives thinning",
          thin["ticker"].nunique() == full["ticker"].nunique())


def test_walk_forward() -> None:
    section("WALK-FORWARD AND CALIBRATION")
    df = big_panel()
    oos = M.walk_forward(df, 21, min_train_years=4)
    check("produced out-of-sample rows", len(oos) > 5000, f"{len(oos)}")

    m = M.evaluate(oos, 21)
    check("metrics computed", bool(m))
    check("independent windows are far fewer than rows",
          m["n_independent"] < m["n_all"] / 10,
          f"{m['n_independent']:,} of {m['n_all']:,}")

    check("coverage is near the target",
          abs(m["coverage"] - 0.5) < 0.08,
          f"{m['coverage']:.3f} - the band should hold half the outcomes")
    check("enough independent windows for a fair test",
          m["n_independent"] > 12000,
          f"{m['n_independent']:,} - below roughly 12,000 the model loses to "
          f"the naive band on pinball loss and the test would be measuring "
          f"the fixture, not the code")
    check("it beats the naive band on pinball loss", m["pinball_gain"] > 0,
          f"model {m['pinball']:.4f} vs naive {m['pinball_naive']:.4f}")

    # The real test, and the one that holds at every sample size: is it
    # calibrated in every bucket, not just on average?
    check("conditional calibration is much better than the naive band",
          m["worst_bucket_error"] < 0.6 * m["worst_bucket_error_naive"],
          f"worst bucket off by {m['worst_bucket_error']:.3f} vs naive "
          f"{m['worst_bucket_error_naive']:.3f} - this is what the model is "
          f"actually for")

    print("\n" + M.summarize(m))


def test_leakage() -> None:
    section("LEAKAGE")
    df = panel()
    X, carry = features.training_matrix(df)
    carry = carry.copy()
    carry["year"] = pd.to_datetime(carry["date"]).dt.year

    # Fit on early years only, then confirm nothing about the fit changes when
    # later data is deleted from the input entirely.
    cut = int(carry["year"].quantile(0.6))
    early = carry["year"] <= cut
    a = M.RangeModel(horizon=21).fit(X.loc[early.values], carry.loc[early, "z"])
    trimmed_df = df[pd.to_datetime(df["date"]).dt.year <= cut]
    Xb, cb = features.training_matrix(trimmed_df)
    b = M.RangeModel(horizon=21).fit(Xb, cb["z"])

    probe = X.loc[early.values].head(500)
    za = a.predict_z(probe).to_numpy()
    zb = b.predict_z(probe).to_numpy()
    check("deleting later years changes nothing",
          np.allclose(za, zb, atol=1e-9),
          "a feature is reaching forward in time")

    # And the forward return must never be knowable from the features.
    fwd = carry.loc[early, "fwd_ret"].to_numpy()
    worst = 0.0
    for c in ("log_rv_21", "rv_slope_long", "range_vol_21", "downside_vol_21"):
        v = X.loc[early.values, c].to_numpy()
        ok = np.isfinite(v) & np.isfinite(fwd)
        if ok.sum() > 1000:
            worst = max(worst, abs(float(np.corrcoef(v[ok], fwd[ok])[0, 1])))
    check("no feature predicts the direction of the move", worst < 0.15,
          f"largest |corr| with the forward return is {worst:.3f} - anything "
          f"high here would mean the target leaked in")


def test_ablation() -> None:
    section("ABLATION HARNESS")
    df = panel()
    groups = {k: v for k, v in features.FEATURE_GROUPS.items()
              if k in ("vol", "shape", "earnings")}
    abl = M.ablation(df, 21, groups)
    check("baseline measured", "vol" in abl)
    check("groups measured against it", len(abl) >= 2)
    print(f"\n  {'group':<10} {'cols':>5} {'pinball':>9} {'delta':>9} {'t':>7}"
          f"   verdict")
    for name, r in abl.items():
        t = r.get("t")
        ts = "      -" if t is None or pd.isna(t) else f"{t:+7.2f}"
        print(f"  {name:<10} {r['columns']:>5} {r['pinball']:>9.4f} "
              f"{r['delta']:>+9.5f} {ts}   {r.get('verdict', '')}")

    # The fixtures plant no relationship between earnings and volatility, so
    # that group must come back as a null. If the harness rewards it anyway,
    # it is rewarding extra columns rather than measuring skill.
    if "earnings" in abl and not pd.isna(abl["earnings"].get("t", np.nan)):
        check("an unrelated group is not credited",
              abl["earnings"]["t"] > -2,
              f"t={abl['earnings']['t']:+.2f} for a feature the fixtures give "
              f"no signal to")


def main() -> int:
    print("Range forecast - offline model checks")
    for fn in (test_features, test_earnings_window, test_quantiles,
               test_pinball, test_non_overlapping, test_walk_forward,
               test_leakage, test_ablation):
        try:
            fn()
        except Exception:  # noqa: BLE001
            global FAIL
            FAIL += 1
            FAILURES.append(f"{fn.__name__} raised")
            print(f"  CRASH in {fn.__name__}")
            traceback.print_exc()

    print(f"\n{'=' * 70}\n{PASS} passed, {FAIL} failed")
    if FAILURES:
        print("\nfailures:")
        for f in FAILURES:
            print(f"  - {f}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
