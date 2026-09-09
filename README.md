# Stock range forecasts

Answers one question, for every stock in the S&P 500, over three horizons:

> Half the time, **AAPL** finishes the next month between **$X** and **$Y**.

The same statement the football models make about a game's margin — and
deliberately the *only* statement this makes. It does not predict direction.

---

## Why the range and not the direction

This is the important idea, so it goes first.

In the football models the band was decoration around a point forecast that
carried real signal: power ratings predict a game's margin at about 0.65
correlation. For a stock over the next few weeks, the point forecast is the
part that **doesn't** work. Expected drift over a month is a fraction of a
percent and is buried in noise; predicting direction is close to hopeless, and
every hour spent on it is wasted.

Dispersion is different. Volatility clusters — a stock that has been swinging
3% a day keeps swinging, a quiet one stays quiet — with strong autocorrelation
at every horizon that matters. So *how far* a stock is likely to move is
forecastable in a way that *which way* is not.

This project therefore keeps the half of the sports models that worked
(calibrated intervals, graded honestly against what happened) and drops the
half that measured as a null (an edge against a market price).

**A calibrated range is not a trading strategy.** It says what is plausible,
not what is mispriced.

---

## Run the probe first

Actions → **Probe** → *Run workflow*. Then send me the output.

Locally:

```
pip install -r requirements.txt
python test_data.py     # 33 offline checks, no network
python selftest.py      # what the live data actually contains
```

The probe reports the usual things — which source answered, how much history,
what's missing — and then three that decide whether this is worth continuing:

**Section 1, survivorship.** Whether index membership history is reachable.
**Section 4, the premise.** Whether trailing volatility actually predicts
future volatility in real data. If that correlation comes back near zero,
nothing downstream can work and the right move is to stop. Both sports builds
discovered their premise was wrong *after* the model was finished; this asks
first.
**Section 5, the bar.** How well the obvious answer — "the next month looks
like the last month" — already does. Any model has to beat that.

---

## The survivorship trap

Take today's S&P 500, run it back twenty years, and you have selected for
companies that survived. The ones that blew up or were dropped in distress are
missing — and those are exactly the names whose volatility exploded before they
left. A model trained on survivors learns that stocks are calmer than they are,
and it is most wrong precisely when it matters.

So membership here is a function of date, not a fixed list. Roughly a thousand
tickers were in the index at some point since 2004 against the 500 in it now,
and that gap is the size of the bias avoided. When the history file can't be
fetched the loader still runs, but it says loudly that everything downstream is
biased, and the probe repeats the warning.

## The adjustment trap

An unadjusted 2-for-1 split reads as a 50% single-day loss. Feed those to a
volatility model and it learns that stocks are twice as wild as they are, and
nothing about that failure is loud. So `prices.audit()` counts the moves too
large to be real, and flags the cluster near log(2) that is the signature of an
unadjusted split. The probe prints a verdict either way.

## The overlap trap

This one has no fix, only honesty. Sample a 30-day forward window every day and
consecutive samples share 29 of their 30 days — two thousand observations might
be sixty independent ones. It is the same problem as "200 comparable games are
not 200 independent games", cubed, and it is how people talk themselves into
believing a volatility model works.

So: evaluation on non-overlapping windows, block bootstrap for error bars, and
walking forward by calendar time only. The probe's own correlations are
computed over overlapping windows and it says so where it prints them.

---

## What gets built next, once the probe reports back

1. **Features.** Trailing realized volatility at 5, 21, 63 and 252 days — the
   HAR structure, which is a stubbornly hard baseline to beat and is this
   project's equivalent of the power ratings. Then the term-structure slope,
   day-range volatility, overnight gaps, dollar volume, market-wide volatility
   and the stock's sensitivity to it, and earnings proximity if the probe finds
   the dates go back far enough.
2. **The model.** Gradient boosting fitted with **pinball loss** directly at
   the 10th, 25th, 50th, 75th and 90th percentiles — one model per quantile per
   horizon. Not a mean and a standard deviation: real returns are fat-tailed
   and left-skewed, so a symmetric band misdescribes both ends and understates
   the downside, which is the end that matters.
3. **Calibration as the headline metric.** Did 50% of outcomes land inside the
   stated middle half? That is the product working or not, and it is checkable
   by anyone.
4. **A page and a tracker**, the same shape as the football ones: today's
   ranges, and a forward-only scoreboard of every range published before its
   window closed.

Each feature group earns its place through walk-forward pinball loss, measured
per group and paired, or it gets reported as a null — the same harness that
caught a real quarterback effect the football build's stacked ablation had
hidden.

---

## Files

| file | what it does |
|---|---|
| `config.py` | every tunable, with the reasoning attached |
| `universe.py` | index membership as a function of date |
| `prices.py` | adjusted daily bars, two sources, returns, realized volatility, adjustment audit |
| `selftest.py` | the live probe |
| `fixtures.py` | a synthetic market with clustered volatility, an unadjusted split, a delisting and a recent listing |
| `test_data.py` | 33 offline checks |

## What the offline tests can and cannot prove

`test_data.py` runs with no network and checks, among other things, that a
trailing volatility window doesn't move when the future is deleted, that a
forward return never borrows the next ticker's prices, that an unadjusted split
is detected, and that volatility persistence is recovered from the fixtures —
*and* that shuffling the returns destroys it, so the measurement has teeth.

What they cannot prove is anything about the real market. That is what the
probe is for.
