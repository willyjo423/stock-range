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

---

## What the model is, and what it needs

Per horizon, five gradient boosters fitted with **pinball loss** at the 10th,
25th, 50th, 75th and 90th percentiles — of the forward return *divided by the
naive band width*. Predicting that ratio rather than the return itself means
the trees only learn how the shape differs from the naive assumption instead of
relearning the level of volatility, which is the same trick as riding the
football model on a rescaled ratings baseline, and it is what lets the model
extrapolate to a stock wilder than anything in its training data.

Two corrections sit on top, both fitted out of sample:

**Width.** Regularised quantile regression is biased toward the conditional
median — shrinkage pulls the 25th up and the 75th down — so the raw band comes
out about 12% too narrow and under-covers. One scalar per quantile, fitted on a
held-back slice, fixes it.

**Order.** Five independent models can produce a 25th above a 75th on an odd
row. Each row is sorted, so a range can never read backwards.

### It needs a large panel, and here is the measurement

Estimating conditional quantiles costs variance; the naive answer costs none.
So below a certain amount of *independent* evidence, the model is worse than
doing nothing. Measured on the fixtures:

| independent windows | pinball vs naive | worst bucket, model / naive |
|---|---|---|
| 5,614 | −2.9% | 6.2 / 9.8 |
| 18,629 | +0.2% | 2.3 / 9.4 |
| 51,840 | +1.1% | 3.2 / 11.0 |

A 40-ticker test would have "proved" the approach does not work. It shows that
40 tickers is not enough — which is exactly why the universe question was worth
answering before writing any of this. The real panel has well past 100,000
independent windows at a one-month horizon.

Note what improves at *every* size: conditional calibration. The naive band's
worst volatility bucket sits 9–11 points from target; the model's sits 2–3.
That is the product, and the pinball gain is the tiebreak.

---

## Where it landed

Second bootstrap, after removing the two harmful feature groups and fixing the
calibration split. 250 index members, 2004-2026:

| horizon | vs naive | coverage | worst bucket | naive's worst | windows |
|---|---|---|---|---|---|
| 1 week | **+5.3%** | 49.8% | **0.5** | 18.6 | 125,789 |
| 1 month | **+1.0%** | 48.5% | **2.7** | 9.5 | 29,857 |
| 3 months | −0.7% | 46.5% | 6.5 | 9.2 | 9,892 |

At one week the conditional calibration is essentially exact — 49.5%, 49.7%,
50.2%, 49.8% across the four volatility quartiles, against a naive band running
31.4% to 63.2%. That 32-point spread closing to half a point is what this
project is for.

Three months still loses, and the reason is visible in the last column: 9,892
independent windows is below the crossover measured on the fixtures, where the
model needs roughly 12,000 before it beats doing nothing. Running the full
~1,000-name universe instead of a 250 sample should roughly quadruple that.
Until it does, that horizon should not ship.

---

## What the first bootstrap found

250 index members, 2004-2026, roughly 780,000 ticker-days per horizon.

**The volatility core works at every horizon.** Measured alone against the
naive band, on non-overlapping windows:

| horizon | vol core | naive | verdict |
|---|---|---|---|
| 1 week | 0.3834 | 0.4015 | +4.5% better |
| 1 month | 0.3086 | 0.3120 | +1.1% better |
| 3 months | 0.2970 | 0.3001 | +1.0% better |

**And the full model lost at two of the three**, because two feature groups
were actively harmful:

```
group      1 week            1 month           3 months
shape      -0.0010 t=-10.8   -0.0008 t= -3.5   -0.0004 t= -0.9
earnings   -0.0031 t=-20.3   -0.0012 t= -5.4   -0.0001 t= -0.2
flow       -0.0000 t= -0.0   +0.0028 t= +7.4   +0.0094 t=+11.3
market     +0.0104 t=+38.9   +0.0117 t=+21.2   +0.0190 t=+20.2
```

Positive is worse. `market` — VIX, index volatility, beta — is the most
harmful thing in the model at every horizon, and for a structural reason: those
values are nearly identical for every stock on a given day, so the trees end up
keyed to particular historical regimes rather than to the stock in front of
them. Train through 2015 and the model has learned what a VIX of 15 implied
then, which is not what it implies in the year being predicted.

`shape` and `earnings` both help, strongly at one week and fading with horizon
— which is what you would expect, since a single earnings report dominates a
five-day window and barely registers over a quarter.

So the shipped feature set is **vol + shape + earnings**. `market` and `flow`
are still built and still ablated every run, so that decision keeps being
re-tested rather than taken on trust.

### The conditional gap, which is the point

At one week the naive band covered **31.4%** of outcomes for the calmest
quarter of stocks and **63.2%** for the wildest — a 32-point spread hiding
behind a respectable-looking overall number. The model brought that to
47.2% / 42.8%. Worst bucket: **7.2 points off target against the naive band's
18.6**.

### Two limitations worth stating

**Survivorship is only half-fixed.** The universe correctly includes the 492
companies that left the index, but yfinance cannot serve most delisted tickers
— 106 failed to download. 250 requested, 158 returned. The bias is reduced, not
eliminated, and the names still missing are the ones that blew up.

**The width calibration had a real bug.** It split the held-back slice by row
position, and the panel is sorted by ticker, so it held back the alphabetically
last tickers instead of the most recent dates. It was measuring in-period,
found nothing to correct, and returned factors of 1.00-1.06 while the finished
bands under-covered by five to eight points at every horizon. It now splits on
date.

---

## The direction arrows, and what they are worth

The page marks each range with an up, down or flat indicator. Two things make
it honest rather than decorative.

**It is measured against the day's cross-section, not against zero.** Every
stock's median forecast sits above today's price over a quarter, because the
market drifts up and the model learned that from twenty years of it. That part
is identical for all five hundred names and nobody can act on it. The arrow
shows each stock's tilt *minus the median tilt across the whole page*, which
leaves only what is specific to the name. Without that subtraction every row
would carry an up arrow.

**Whether even that carries information is measured, not assumed.** Every
bootstrap now reports:

```
Direction, which this model does not claim to predict:
  called the right way   xx.x%
  always saying up       xx.x%
  -> skill +x.x points (t = +x.x) on nn,nnn windows
  against the day's own cross-section: xx.x% (t = +x.x)
```

The base rate matters more than the hit rate. Equities rise more often than
they fall, so a model that always says "up" scores well above 50% while knowing
nothing at all. Skill is the gap between the two, and the t says whether that
gap survives.

**If it comes back near zero, take the arrows off the page.** The features are
all measures of how far a stock moves, not which way; nothing in the design
gives it directional information, and a faint arrow that means nothing is worse
than no arrow because it invites exactly the reading the rest of the page is
careful to avoid. On the fixtures — which contain no directional signal by
construction — it correctly reports 50.0% against a 50.1% base rate at t = −0.4.

---

# Options flow

A second thing built on the same universe and, more importantly, on the same
bands. Short-dated, at the money, large premium, into a contract that barely
existed yesterday.

## What free data can and cannot do

The four criteria split cleanly, and the split is the whole design.

| criterion | free data | fidelity |
|---|---|---|
| expiration inside two weeks | exact | full |
| at the money | exact | full |
| low open interest | exact, and better than exact | full |
| $50,000+ premium | volume x mid x 100 | **approximate** |

Open interest is the strong one. It settles overnight and does **not** tick
during the session, so the figure in a chain pulled at 11am is yesterday's.
Today's volume against it is a real before-and-after, and "more contracts
changed hands today than existed yesterday" is about as close to *new
positioning* as anything gets without a tape.

Premium is the weak one, and there is no fixing it. On a tape, $50k of premium
is one trade's size times its price. Here it is everything that traded in that
contract, however many hands it took. One $80,000 sweep and eight hundred $100
retail lots produce the same number.

## The one lever against that: time

Scan several times a session and difference the cumulative volume. A block
lands almost entirely inside one interval; a dribble spreads across all of
them. That share is `burst`, and a contract that bursts near 1.0 on meaningful
premium is the closest free data gets to *one order did that*.

It needs at least two snapshots in a day, and reports `None` until it has
them - not 1.0, which would claim concentration that was never measured. This
is the reason the workflow runs three times a session rather than once after
the close.

`test_flow.py` plants the two side by side: `BLOCK` trades its whole 900 lots
in the afternoon interval, `DRIBBLE` trades 450 in each. Identical premium,
bursts of 1.0 and 0.5.

## What is deliberately not attempted

**Bought or sold.** That needs the quote standing at the instant of the trade.
What is available is the last print against the current market, which is a fair
proxy when the trade just happened and meaningless when it happened at 10am. So
it is computed, and marked `firm` only when the interval's own volume proves
the trade fell inside it. Otherwise it reads `unknown`.

**Opening versus closing intent.** Dropped from the brief. But next morning's
open interest answers it a day late for free - if OI rose by roughly what
traded, the position was opened and held - so every flag gets that stamp on the
following run, and the grader checks whether the stamp separates anything.

**Anything called bullish or bearish.** Call premium is call premium. Someone
sold every one of those contracts.

## The six decoys

The easy test is that a screen flags the planted block. The useful test is that
it declines to flag six things that look like it in exactly one respect:

| decoy | identical to the block except | must fail on |
|---|---|---|
| `DRIBBLE` | 50,000 open interest | new positioning |
| `FARDATE` | thirty days out | expiry |
| `OTM` | 25% out of the money | moneyness |
| `WIDE` | quoted 0.02 / 2.00 | quote quality |
| `SMALL` | a tenth the size | premium |
| `QUIET` | never trades | premium |

And each gate is checked for doing work: loosening it must let its decoy
through. A gate that changes nothing when loosened was never catching anything.

## How the scan gets graded, and why the band is the yardstick

This is the part that makes it worth running rather than worth looking at.

Every "unusual options activity" study has the same hole: flagged names skew
toward volatile names, volatile names move more, so any screen looks prescient
if movement is measured in percent. The fix here was already built - the range
model states, per stock, where it lands half the time over the next week,
conditional on how volatile that stock already is.

So the question has an exact form:

> Do flagged names leave **their own** middle-half band more often than
> unflagged names on the same days?

A band is 50/50 by construction, so a wild name gets no credit for being wild.
`test_flow.py` builds the trap directly: flagged names drawn only from the
high-volatility group, moving four times as far in raw terms, and checks that
the grader reports no effect - which it must, because there is none.

Three readings, in increasing order of power: breakout rate (binary, weakest),
move size in band half-widths (continuous, extracts far more from a small
archive), and direction for one-sided flags only. Direction is the one most
likely to be a null and it is reported against the unflagged base rate rather
than against 50%.

Every result prints its own power alongside it, because *no effect detected*
and *not enough data to detect one* read identically and mean opposite things.
An eight-point effect needs roughly 156 flagged observations before it clears
two standard errors; until then the grader says `NOT ENOUGH YET` instead of
printing a verdict.

**And if it stays flat as the sample grows, the honest move is to stop running
the scan** - not to loosen the filters until something looks significant.

## No lookahead

A flag is raised during the session on day D from a delayed chain. The band it
is graded against is anchored to D's *close* and covers the week after it. So
whatever the flow already did to the price on day D sits inside the anchor, not
inside the outcome. The test is conservative on purpose.

## Run the probe first

Actions -> **Flow probe** -> *Run workflow*. Twice, an hour apart, during US
market hours (13:30-20:00 UTC). Then send me the output.

Nothing in `chains.py` was written with a live endpoint in front of it, and
that is exactly the situation that produced four separate bugs in the football
builds. The probe answers five things:

1. Which source answers, and how long 500 names would take.
2. Whether the fields are there, and whether contract symbols agree with their
   own stated strikes and expiries.
3. **How many contracts survive each filter.** If the whole index would flag
   three thousand names a day the filters are decoration; if it would flag zero
   they can never be graded. Both are findable now rather than in a month.
4. Whether open interest really does hold still during the session. This is
   what the second run answers, and the new-positioning filter rests on it.
5. What a day of scanning costs in requests and minutes.

## Files

| file | what it does |
|---|---|
| `chains.py` | option chains from CBOE and yfinance, normalised; OCC symbol parsing; a quality check |
| `flow.py` | enrichment, snapshot differencing, the four gates, scoring, the day's flag file |
| `flow_scan.py` | one scan, end to end - the thing the schedule runs |
| `flow_grade.py` | forward grading against the range model's own bands |
| `flow_dashboard.py` | `docs/flow.html`, with the data's limits printed on the page |
| `flow_probe.py` | the live probe - run this first |
| `flow_fixtures.py` | a synthetic chain with a planted block and six decoys |
| `test_flow.py` | 81 offline checks |

## If free turns out not to be enough

The probe's section 3 is what decides it. If short-dated ATM premium is so
diffuse that nothing separates a block from background noise, the missing piece
is the tape, and the tape is a paid feed - a per-trade options API, or a retail
flow tool. That is a real answer and worth reaching honestly rather than by
building a screen that fires often enough to feel useful. The grader is what
tells the difference.
