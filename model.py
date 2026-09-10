"""Quantile models, and the measurements that decide whether they are any good.

One model per quantile per horizon, fitted with pinball loss directly on the
standardized forward return. Not a mean and a standard deviation: the probe
showed real returns are fat-tailed and left-skewed, so a symmetric band
misdescribes both ends and understates the downside, which is the end that
matters.

What "good" means here is not obvious, and getting it wrong is the main way a
volatility model fools its author.

* **Coverage** - did 50% of outcomes land inside the stated middle half. The
  probe found the naive band already does this overall, so hitting 50% is not
  an achievement, it is the entry fee.
* **Conditional coverage** - is it 50% for calm stocks *and* wild ones. The
  naive band is not: 31% for the calmest quarter at a one-week horizon and 64%
  for the wildest. Closing that is the actual job.
* **Sharpness** - how wide the band is. Any band can reach 50% coverage by
  being wide on some days and narrow on others at random; the useful model is
  the one that is narrow when it can afford to be.
* **Pinball loss** - the one number that trades those off correctly, and the
  one the model is fitted on. It is the objective, and the others explain it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from features import FEATURE_COLUMNS

log = logging.getLogger(__name__)


class FeatureMismatchError(RuntimeError):
    """Saved model expects features this code no longer produces."""


def _regressor(quantile: float, **kw) -> HistGradientBoostingRegressor:
    params = dict(
        loss="quantile", quantile=quantile,
        max_iter=300, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=120,
        l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
        random_state=config.RANDOM_SEED,
    )
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """The loss a quantile model is actually judged on.

    Under-predicting the 90th percentile is cheap and over-predicting it is
    expensive, in exactly the ratio that makes the minimiser the true 90th
    percentile. Averaging it across quantiles scores the whole distribution.
    """
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


@dataclass
class RangeModel:
    """Predicts a distribution of standardized returns, then rescales it."""

    horizon: int = 21
    quantiles: list = field(default_factory=lambda: list(config.QUANTILES))
    models: dict = field(default_factory=dict)
    features: list = field(default_factory=lambda: list(FEATURE_COLUMNS))
    spread: dict = field(default_factory=dict)
    trained_rows: int = 0
    trained_range: tuple = ()

    def fit(self, X: pd.DataFrame, y: pd.Series,
            dates: pd.Series | None = None,
            calibrate: bool = True) -> "RangeModel":
        X = X[self.features]
        z = np.asarray(y, dtype=float)

        # Hold back the most recent DATES before fitting, and use them only to
        # correct the width of the finished band.
        #
        # The first version of this split by row position, which was a real
        # bug: the panel is sorted by ticker then date, so "the last 15% of
        # rows" was the alphabetically last tickers, not the most recent
        # period. The calibration was therefore measured in-period, found
        # nothing to fix, and came back at 1.00-1.06 - while the walk-forward
        # showed the finished bands under-covering by five to eight points at
        # every horizon. It was correcting the wrong thing on the wrong sample.
        n = len(X)
        holdout = None
        if calibrate and n > 20000:
            if dates is not None:
                d = pd.Series(pd.to_datetime(pd.Series(dates).to_numpy()))
                cutoff = d.quantile(0.85)
                holdout = (d > cutoff).to_numpy()
                if holdout.sum() < 2000 or (~holdout).sum() < 10000:
                    holdout = None
            else:
                log.warning("no dates given, so the width calibration falls "
                            "back to a positional split - fine for a fixture, "
                            "wrong for a panel sorted by ticker")
                cut = int(n * 0.85)
                holdout = np.zeros(n, dtype=bool)
                holdout[cut:] = True

        train = slice(None) if holdout is None else ~holdout
        Xf = X.loc[train] if holdout is not None else X
        zf = z[train] if holdout is not None else z

        for q in self.quantiles:
            self.models[q] = _regressor(q).fit(Xf, zf)
        self.trained_rows = int(len(Xf))

        self.spread = {q: 1.0 for q in self.quantiles}
        if holdout is not None:
            self._fit_spread(X.loc[holdout], z[holdout])
        log.info("fitted %d quantile models on %d rows", len(self.models),
                 self.trained_rows)
        return self

    # -- width calibration ---------------------------------------------------
    def _fit_spread(self, X: pd.DataFrame, z: np.ndarray) -> None:
        """Correct how wide the finished band is, on data the trees never saw.

        Regularised quantile regression is biased toward the conditional
        median: shrinkage pulls the 25th up and the 75th down, so the band
        comes out systematically too narrow and under-covers. On the fixtures
        that cost 12% of the width and turned a model that improved conditional
        calibration into one that lost to the naive band outright.

        So each quantile gets one scalar, stretching or shrinking its distance
        from the predicted median, chosen to minimise pinball loss on held-out
        rows. One parameter per quantile, fitted out of sample, and it cannot
        reorder anything because the factors are positive and the result is
        sorted anyway. It is the same correction the football model applies to
        its finished margin, for the same reason.
        """
        raw = pd.DataFrame({q: self.models[q].predict(X) for q in self.quantiles},
                           index=X.index)
        mid = raw[0.5].to_numpy() if 0.5 in raw.columns else raw.mean(axis=1).to_numpy()
        grid = np.arange(0.6, 2.01, 0.02)
        for q in self.quantiles:
            if q == 0.5:
                self.spread[q] = 1.0
                continue
            base = raw[q].to_numpy() - mid
            losses = [pinball(z, mid + lam * base, q) for lam in grid]
            self.spread[q] = float(grid[int(np.argmin(losses))])
        widened = [f"{q:.2f}:{self.spread[q]:.2f}" for q in self.quantiles]
        log.info("spread calibration (held-out): %s", ", ".join(widened))

    def check_compatible(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise FeatureMismatchError(
                f"The saved model expects features this code no longer "
                f"produces: {missing}.\n\nFix: re-run the Bootstrap workflow "
                f"to retrain.")

    def predict_z(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standardized quantiles, forced into order.

        Each quantile is a separate model, so nothing stops the 25th coming
        back above the 75th on an odd row - "quantile crossing", and it makes
        a range that reads backwards. Sorting each row fixes it, and it is
        worth doing loudly rather than hoping it never happens.
        """
        self.check_compatible(X)
        Xf = X[self.features]
        raw = pd.DataFrame(
            {q: self.models[q].predict(Xf) for q in self.quantiles},
            index=X.index)
        mid = (raw[0.5].to_numpy() if 0.5 in raw.columns
               else raw.mean(axis=1).to_numpy())
        out = pd.DataFrame(index=X.index)
        for q in self.quantiles:
            lam = self.spread.get(q, 1.0)
            out[q] = mid + lam * (raw[q].to_numpy() - mid)
        values = np.sort(out.to_numpy(), axis=1)
        return pd.DataFrame(values, columns=self.quantiles, index=X.index)

    def predict(self, X: pd.DataFrame, scale: pd.Series,
                close: pd.Series | None = None) -> pd.DataFrame:
        """Return quantiles, and price levels when a close is supplied."""
        z = self.predict_z(X)
        s = pd.to_numeric(scale, errors="coerce").to_numpy()
        out = pd.DataFrame(index=X.index)
        for q in self.quantiles:
            out[f"ret_q{int(q * 100)}"] = z[q].to_numpy() * s
        if close is not None:
            c = pd.to_numeric(close, errors="coerce").to_numpy()
            for q in self.quantiles:
                out[f"px_q{int(q * 100)}"] = c * np.exp(
                    out[f"ret_q{int(q * 100)}"].to_numpy())
        return out


# ------------------------------------------------------------- baselines
def naive_z(quantiles: list) -> dict:
    """The obvious answer, expressed in the model's own units.

    Assume the next window looks like the trailing one and that returns are
    normal. In standardized terms that is just the normal quantiles - which
    makes the comparison exact rather than approximate.
    """
    return {q: float(norm.ppf(q)) for q in quantiles}


# ------------------------------------------------------------ evaluation
def non_overlapping(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Thin the rows so evaluation windows do not share days.

    Sampling a 21-day forward window every day means consecutive rows share 20
    of their 21 days. Two thousand of those are not two thousand observations,
    and treating them as such is how a volatility model gets talked up. Keeping
    every horizon-th row per ticker makes the windows disjoint.
    """
    out = []
    for _, block in df.sort_values(["ticker", "date"]).groupby("ticker",
                                                              sort=False):
        out.append(block.iloc[::horizon])
    return pd.concat(out) if out else df.iloc[0:0]


def walk_forward(panel: pd.DataFrame, horizon: int,
                 columns: list | None = None,
                 min_train_years: int = 5,
                 max_years: int | None = None) -> pd.DataFrame:
    """Refit each year on everything before it, predict that year only.

    `max_years` tests only the most recent N years while still training on
    everything before each of them. It exists for the ablation, where the
    question is which feature group is better rather than what the model
    scores, and eighteen refits per group is most of the bootstrap's runtime.
    """
    from features import training_matrix

    X_all, carry = training_matrix(panel, columns=columns)
    carry = carry.copy()
    carry["year"] = pd.to_datetime(carry["date"]).dt.year
    years = sorted(carry["year"].unique())
    testable = years[min_train_years:]
    if max_years is not None and len(testable) > max_years:
        testable = testable[-max_years:]
    keep = set(testable)

    out = []
    for i, yr in enumerate(years):
        if yr not in keep:
            continue
        # A forward window straddles the year boundary, so training rows whose
        # own window reaches into the test year would be peeking. Dropping the
        # last `horizon` rows before the cut removes exactly that overlap.
        cut = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=horizon * 2)
        train = pd.to_datetime(carry["date"]) < cut
        test = carry["year"] == yr
        if train.sum() < 5000 or test.sum() < 200:
            continue

        m = RangeModel(horizon=horizon, features=list(X_all.columns))
        m.fit(X_all.loc[train.values], carry.loc[train, "z"],
              dates=carry.loc[train, "date"])
        z = m.predict_z(X_all.loc[test.values])

        block = carry.loc[test].copy()
        for q in m.quantiles:
            block[f"z_q{int(q * 100)}"] = z[q].to_numpy()
        block["year_tested"] = yr
        out.append(block)
        log.info("walk-forward %s h=%d: trained %d, tested %d",
                 yr, horizon, int(train.sum()), int(test.sum()))

    return pd.concat(out) if out else pd.DataFrame()


def evaluate(oos: pd.DataFrame, horizon: int,
             quantiles: list | None = None) -> dict:
    """Everything worth knowing, including what the naive answer would score."""
    if oos.empty:
        return {}
    quantiles = quantiles or list(config.QUANTILES)
    d = non_overlapping(oos, horizon)
    if d.empty:
        return {}

    z = d["z"].to_numpy(dtype=float)
    nz = naive_z(quantiles)

    m = {"n_all": int(len(oos)), "n_independent": int(len(d)),
         "horizon": horizon}

    model_loss, naive_loss = [], []
    for q in quantiles:
        pred = d[f"z_q{int(q * 100)}"].to_numpy(dtype=float)
        model_loss.append(pinball(z, pred, q))
        naive_loss.append(pinball(z, np.full(len(z), nz[q]), q))
    m["pinball"] = float(np.mean(model_loss))
    m["pinball_naive"] = float(np.mean(naive_loss))
    m["pinball_gain"] = m["pinball_naive"] - m["pinball"]
    m["pinball_gain_pct"] = (100.0 * m["pinball_gain"] / m["pinball_naive"]
                             if m["pinball_naive"] else float("nan"))

    lo, hi = config.BAND
    lo_c, hi_c = f"z_q{int(lo * 100)}", f"z_q{int(hi * 100)}"
    inside = (z >= d[lo_c].to_numpy()) & (z <= d[hi_c].to_numpy())
    m["coverage"] = float(inside.mean())
    m["coverage_target"] = hi - lo
    width = (d[hi_c] - d[lo_c]).to_numpy()
    m["median_width_z"] = float(np.median(width))

    naive_inside = (z >= nz[lo]) & (z <= nz[hi])
    m["coverage_naive"] = float(naive_inside.mean())
    m["median_width_naive_z"] = float(nz[hi] - nz[lo])

    # The measurement that matters: is it calibrated everywhere, or only on
    # average? The naive band's failure is invisible in the headline number.
    d = d.copy()
    d["_inside"] = inside
    d["_width"] = width
    d["_naive_inside"] = naive_inside
    try:
        d["_bucket"] = pd.qcut(d["rv_ref"], 4,
                               labels=["calmest", "calm", "active", "wildest"],
                               duplicates="drop")
    except (ValueError, KeyError):
        d["_bucket"] = "all"

    m["by_volatility"] = {
        str(name): {
            "n": int(len(b)),
            "coverage": float(b["_inside"].mean()),
            "coverage_naive": float(b["_naive_inside"].mean()),
            "median_width_z": float(b["_width"].median()),
        }
        for name, b in d.groupby("_bucket", observed=True)
    }
    spread = [v["coverage"] for v in m["by_volatility"].values()]
    naive_spread = [v["coverage_naive"] for v in m["by_volatility"].values()]
    # One number for "is it calibrated everywhere": the worst bucket's distance
    # from target. The naive band scores badly here and well on the headline.
    m["worst_bucket_error"] = float(max(abs(c - m["coverage_target"])
                                        for c in spread)) if spread else None
    m["worst_bucket_error_naive"] = float(
        max(abs(c - m["coverage_target"]) for c in naive_spread)) if naive_spread else None

    if "year_tested" in d.columns:
        m["by_year"] = {
            int(y): {"n": int(len(b)), "coverage": float(b["_inside"].mean())}
            for y, b in d.groupby("year_tested")}
    return m


def summarize(m: dict) -> str:
    if not m:
        return "No metrics."
    lines = [
        f"Independent windows      : {m['n_independent']:,} "
        f"(of {m['n_all']:,} overlapping rows)",
        f"Pinball loss             : {m['pinball']:.4f}",
        f"  naive band             : {m['pinball_naive']:.4f}",
        f"  -> {m['pinball_gain_pct']:+.1f}% "
        f"{'better' if m['pinball_gain'] > 0 else 'WORSE'} than assuming "
        f"next month looks like last",
        "",
        f"Middle-half coverage     : {m['coverage'] * 100:.1f}% "
        f"(target {m['coverage_target'] * 100:.0f}%)",
        f"  naive band             : {m['coverage_naive'] * 100:.1f}%",
        f"Median band width        : {m['median_width_z']:.2f} "
        f"(naive {m['median_width_naive_z']:.2f}, in units of the naive band)",
    ]
    if m.get("by_volatility"):
        lines += ["", "Coverage by how volatile the stock already is:",
                  f"  {'bucket':<9} {'model':>7} {'naive':>7} {'width':>7} {'n':>8}"]
        for name, b in m["by_volatility"].items():
            lines.append(f"  {name:<9} {b['coverage'] * 100:6.1f}% "
                         f"{b['coverage_naive'] * 100:6.1f}% "
                         f"{b['median_width_z']:7.2f} {b['n']:8,}")
        lines.append(
            f"  worst bucket is {m['worst_bucket_error'] * 100:.1f} points from "
            f"target (naive: {m['worst_bucket_error_naive'] * 100:.1f})")
    return "\n".join(lines)


def _paired_pinball(a: pd.DataFrame, b: pd.DataFrame, horizon: int,
                    quantiles: list) -> tuple[float, float, int]:
    """Per-window loss difference and its t statistic.

    Paired, because both runs cover the same windows, and on non-overlapping
    windows only - so the sample size in the t is the real one.
    """
    da, db = non_overlapping(a, horizon), non_overlapping(b, horizon)
    join = da.join(db, how="inner", lsuffix="_a", rsuffix="_b")
    if len(join) < 50:
        return float("nan"), float("nan"), len(join)
    z = join["z_a"].to_numpy(dtype=float)
    per = np.zeros(len(join))
    for q in quantiles:
        c = f"z_q{int(q * 100)}"
        pa, pb = join[f"{c}_a"].to_numpy(), join[f"{c}_b"].to_numpy()
        da_ = z - pa
        db_ = z - pb
        per += (np.maximum(q * da_, (q - 1) * da_)
                - np.maximum(q * db_, (q - 1) * db_))
    per /= len(quantiles)
    se = per.std(ddof=1) / np.sqrt(len(per))
    return float(per.mean()), float(per.mean() / se if se else np.nan), len(per)


def thin_rows(panel: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Keep every `stride`-th row per ticker.

    Available, and off by default, because measuring it showed it does not
    work. On a fixture where the shape features genuinely help:

        stride 1   111s   delta -0.00268  t = -2.16   (helps)
        stride 3    63s   delta -0.00031  t = -0.18   (null)
        stride 5    31s   delta +0.00016  t = +0.08   (null)

    It is nearly four times faster and it cannot see the effect any more -
    thinning takes rows out of the training set, which weakens the model, and
    out of the test set, which widens the error bars. A cheap ablation that
    reports every feature group as a null is worse than no ablation, because
    it looks like an answer.

    Trimming years was measured too and saved only 17%, since the years it
    drops are the cheap early ones. The honest saving is running the ablation
    for one horizon instead of three: same measurement, asked once.
    """
    if stride is None or stride <= 1:
        return panel
    out = [b.iloc[::stride] for _, b in
           panel.sort_values(["ticker", "date"]).groupby("ticker", sort=False)]
    return pd.concat(out) if out else panel


def ablation(panel: pd.DataFrame, horizon: int, groups: dict,
             base_group: str = "vol", max_years: int | None = None,
             stride: int = 1) -> dict:
    """What each feature group is worth, measured one at a time.

    Groups are measured alone against the baseline rather than stacked. The NFL
    build's stacked ablation reported its quarterback features at -0.041 points
    - indistinguishable from nothing - when measured alone they were worth
    -0.16 at t = -3.5, because two null groups in front were diluting them.
    """
    results = {}
    base_cols = list(groups[base_group])
    panel = thin_rows(panel, stride)
    base = walk_forward(panel, horizon, columns=base_cols,
                        max_years=max_years)
    if base.empty:
        return {}
    qs = list(config.QUANTILES)
    bm = evaluate(base, horizon)
    results[base_group] = {"columns": len(base_cols), "pinball": bm["pinball"],
                           "delta": 0.0, "t": 0.0, "verdict": "baseline",
                           "coverage": bm["coverage"],
                           "worst_bucket_error": bm["worst_bucket_error"]}

    for name, cols in groups.items():
        if name == base_group:
            continue
        use = base_cols + [c for c in cols if c not in base_cols]
        run = walk_forward(panel, horizon, columns=use,
                           max_years=max_years)
        if run.empty:
            continue
        delta, t, n = _paired_pinball(run, base, horizon, qs)
        gm = evaluate(run, horizon)
        results[name] = {
            "columns": len(use), "pinball": gm["pinball"], "delta": delta,
            "t": t, "n": n, "coverage": gm["coverage"],
            "worst_bucket_error": gm["worst_bucket_error"],
            "verdict": ("helps" if t <= -2 else
                        "HURTS" if t >= 2 else "not distinguishable")}
        log.info("ablation +%-10s pinball %.4f (%+.5f, t=%+.2f) %s",
                 name, gm["pinball"], delta, t, results[name]["verdict"])
    return results


def save_metrics(metrics: dict, path) -> None:
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
