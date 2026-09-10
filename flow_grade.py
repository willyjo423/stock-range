"""Does a flag mean anything? Graded forward, against the model's own bands.

This is the part that makes the scan worth running rather than worth looking
at. Free flow data is a screen; a screen nobody grades is a horoscope with
strike prices.

The comparison, and why it is a fair one
----------------------------------------
The range model already says, for every stock, where it lands half the time
over the next week. That band is calibrated per stock and conditional on how
volatile that stock already is - which solves the problem that ruins most
"unusual options activity" studies. Flagged names skew toward high-volatility
names, so measuring raw movement makes any screen look prescient. Here the
yardstick is each stock's *own* band, so a wild name is not credited for being
wild. Exiting the band is a 50/50 event by construction.

So the question has an exact form: **do flagged names exit their own middle-half
band more often than unflagged names on the same days?** If flagged names break
out 58% of the time and everything else breaks out 50%, that is a real effect
and it is measured, not asserted. If both sit at 50%, the scan sees nothing,
and finding that out in two months is the whole point of building it this way.

Three readings, in increasing order of power
--------------------------------------------
1. **Breakout rate.** Binary, easy to state, weakest at small samples.
2. **Move size.** The realised move in band-half-widths. Continuous, so it
   extracts far more from the same handful of observations - which matters,
   because the archive starts empty and stays small for months.
3. **Direction.** Only for flags that are one-sided and not obviously a
   straddle. Call-tilted flags should break out upward more than the base rate
   if the flow is informed. This is the claim most likely to be a null, and it
   is reported against the unflagged base rate rather than against 50%.

There is no lookahead here. A flag is raised during the session on day D from a
delayed chain; the band is anchored to D's close and covers the week after it.
Whatever the flow already did to the price on day D is inside the anchor, not
inside the outcome.

    python flow_grade.py
    python flow_grade.py --write
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys

import numpy as np
import pandas as pd

import config
import track

log = logging.getLogger(__name__)

# Below this many flagged observations the answer is "not yet", and saying so
# is more useful than a t-statistic computed on nine rows. See `power()`.
MIN_FLAGGED = 60


# ------------------------------------------------------------------ inputs
def load_flag_rows() -> pd.DataFrame:
    """Every archived flag, one row per ticker per day."""
    rows = []
    for path in sorted(config.FLOW_FLAGS.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("skipping %s: %s", path.name, exc)
            continue
        day = payload.get("asof")
        for rec in payload.get("tickers") or []:
            contracts = rec.get("contracts") or []
            confirmed = [c.get("oi_confirmed") for c in contracts
                         if c.get("oi_confirmed") is not None]
            rows.append({
                "asof": pd.Timestamp(day),
                "ticker": rec.get("ticker"),
                "flag_score": rec.get("score"),
                "premium": rec.get("premium"),
                "tilt": rec.get("tilt"),
                "call_share": rec.get("call_share"),
                "multileg": rec.get("looks_multileg", 0),
                "times_flagged": rec.get("times_flagged", 1),
                "nearest_dte": rec.get("nearest_dte"),
                "oi_confirmed": (float(np.mean(confirmed))
                                 if confirmed else np.nan),
            })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).dropna(subset=["ticker"])
              .drop_duplicates(["asof", "ticker"]).reset_index(drop=True))


def build(horizon: str | None = None) -> pd.DataFrame:
    """Graded ranges for the flow horizon, with the flags joined on.

    Every stock that had a published band on a day the scan ran is in here.
    The unflagged ones are the control group and they are the reason any of
    this is interpretable - without them the only comparison available is
    against 50%, which measures the range model as much as it measures the
    scan.
    """
    horizon = horizon or config.FLOW_GRADE_HORIZON
    fc = track.load_forecasts()
    if fc.empty:
        return pd.DataFrame()
    fc = fc[fc["horizon"] == horizon]
    graded = track.grade(fc)
    if graded is None or graded.empty:
        return pd.DataFrame()

    flags = load_flag_rows()
    if flags.empty:
        return pd.DataFrame()

    # Only days the scan actually ran can contribute a control group. On any
    # other day every name looks unflagged for the trivial reason that nothing
    # was looking, and mixing those in dilutes the comparison toward nothing.
    scan_days = set(flags["asof"].dt.normalize())
    graded["asof"] = pd.to_datetime(graded["asof"]).dt.normalize()
    graded = graded[graded["asof"].isin(scan_days)]
    if graded.empty:
        return pd.DataFrame()

    df = graded.merge(flags, on=["asof", "ticker"], how="left")
    df["flagged"] = df["flag_score"].notna().astype(int)

    # The move, in units of the band's own half width. Log terms, because the
    # band is drawn from a return distribution and a linear reading would make
    # every high-priced stock look calmer than it is.
    lo = np.log(pd.to_numeric(df["low"], errors="coerce"))
    hi = np.log(pd.to_numeric(df["high"], errors="coerce"))
    settled = np.log(pd.to_numeric(df["settled"], errors="coerce"))
    centre = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    df["move_z"] = np.where(half > 0, (settled - centre) / half, np.nan)
    df["abs_move_z"] = df["move_z"].abs()
    df["breakout"] = (~df["inside"].astype(bool)).astype(int)
    return df


# -------------------------------------------------------------- statistics
def _two_proportion(a: pd.Series, b: pd.Series) -> dict:
    """Flagged rate against unflagged rate, pooled standard error."""
    na, nb = int(a.notna().sum()), int(b.notna().sum())
    if na < 2 or nb < 2:
        return {}
    pa, pb = float(a.mean()), float(b.mean())
    pooled = (a.sum() + b.sum()) / (na + nb)
    se = math.sqrt(max(pooled * (1 - pooled) * (1 / na + 1 / nb), 1e-12))
    return {"flagged": round(pa, 4), "control": round(pb, 4),
            "gap": round(pa - pb, 4), "se": round(se, 4),
            "z": round((pa - pb) / se, 2), "n_flagged": na, "n_control": nb}


def _welch(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3:
        return {}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(max(va / len(a) + vb / len(b), 1e-12))
    return {"flagged": round(float(a.mean()), 4),
            "control": round(float(b.mean()), 4),
            "gap": round(float(a.mean() - b.mean()), 4),
            "t": round(float((a.mean() - b.mean()) / se), 2),
            "n_flagged": int(len(a)), "n_control": int(len(b))}


def power(n_flagged: int, effect: float = 0.08) -> dict:
    """How much evidence would be needed to see an effect of this size.

    Printed beside every result, because "no effect detected" and "not enough
    data to detect one" read identically and mean opposite things. Both sports
    builds had a version of this confusion in them; here it gets a line of
    output.

    For a breakout rate near 50% against a large control group, the standard
    error is about sqrt(0.25/n), so an eight-point effect needs roughly
    1/0.08^2 = 156 flagged observations before it clears two standard errors.
    """
    need = int(math.ceil(0.25 * 4 / (effect ** 2)))
    return {"have": int(n_flagged), "need_for_effect": need,
            "effect": effect,
            "detectable_now": (round(2 * math.sqrt(0.25 / n_flagged), 3)
                               if n_flagged > 0 else None)}


def summarise(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    flagged = df[df["flagged"] == 1]
    control = df[df["flagged"] == 0]

    out = {
        "n": int(len(df)),
        "n_flagged": int(len(flagged)),
        "n_control": int(len(control)),
        "days": int(df["asof"].nunique()),
        "first": str(df["asof"].min().date()),
        "last": str(df["settle_date"].max().date()),
        "horizon": config.FLOW_GRADE_HORIZON,
        "breakout": _two_proportion(flagged["breakout"], control["breakout"]),
        "move_size": _welch(flagged["abs_move_z"], control["abs_move_z"]),
        "power": power(len(flagged)),
        "enough": bool(len(flagged) >= MIN_FLAGGED),
    }

    # Direction, and only for the flags that are actually one-sided. A
    # two-sided flag or a straddle has no direction to be right about, and
    # including it adds noise to the one test most likely to be a null.
    for side, want in (("calls", 1), ("puts", 0)):
        sub = flagged[(flagged["tilt"] == side) & (flagged["multileg"] == 0)]
        if len(sub) < 10:
            continue
        col = "above" if want else "below"
        out.setdefault("direction", {})[side] = _two_proportion(
            sub[col].astype(int), control[col].astype(int))

    # Whether the loud flags do better than the quiet ones. If the score
    # orders nothing, it should be replaced by a plain filter.
    if len(flagged) >= 40:
        try:
            b = flagged.copy()
            b["bucket"] = pd.qcut(b["flag_score"], 2,
                                  labels=["quieter", "louder"],
                                  duplicates="drop")
            out["by_score"] = {
                str(k): {"n": int(len(g)),
                         "breakout": round(float(g["breakout"].mean()), 4),
                         "abs_move_z": round(float(g["abs_move_z"].mean()), 3)}
                for k, g in b.groupby("bucket", observed=True)}
        except ValueError:
            pass

    # And whether next-morning open interest confirmation separates the real
    # opens from the noise. This is the free stand-in for the trade-level
    # open/close flag, so it is worth knowing if it earns its keep.
    conf = flagged.dropna(subset=["oi_confirmed"])
    if len(conf) >= 40:
        hi = conf[conf["oi_confirmed"] >= 0.5]
        lo = conf[conf["oi_confirmed"] < 0.5]
        if len(hi) >= 10 and len(lo) >= 10:
            out["by_oi_confirmation"] = _two_proportion(hi["breakout"],
                                                        lo["breakout"])
    return out


# ------------------------------------------------------------------ report
def report(s: dict) -> str:
    if not s:
        return ("No graded flags yet. They appear a week after the first scan "
                "that raised one, because the band being graded has to close "
                "first. Nothing here can be hurried without cheating.")

    L = [
        "OPTIONS FLOW, GRADED FORWARD",
        "=" * 68,
        f"Scan days        : {s['days']}  ({s['first']} to {s['last']})",
        f"Names with bands : {s['n']:,}",
        f"  of which flagged: {s['n_flagged']:,}",
        "",
        f"Question: do flagged names leave their own {s['horizon']} band more",
        "often than everything else on the same days? A band is 50/50 by",
        "construction, so the control group is the number that matters.",
        "",
    ]

    b = s.get("breakout") or {}
    if b:
        L += [
            f"Left the band    flagged {b['flagged'] * 100:5.1f}%   "
            f"others {b['control'] * 100:5.1f}%   "
            f"gap {b['gap'] * 100:+5.1f} pts  (z = {b['z']:+.2f})",
        ]
    m = s.get("move_size") or {}
    if m:
        L += [
            f"Move size        flagged {m['flagged']:5.2f}    "
            f"others {m['control']:5.2f}    "
            f"gap {m['gap']:+5.2f}      (t = {m['t']:+.2f})",
            "                 in band half-widths; 1.0 means it finished at "
            "the edge",
        ]

    d = s.get("direction") or {}
    if d:
        L += ["", "Direction, for one-sided flags only:"]
        for side, r in d.items():
            if not r:
                continue
            L.append(f"  {side:<6} broke {'up' if side == 'calls' else 'down'}"
                     f"  {r['flagged'] * 100:5.1f}%  vs "
                     f"{r['control'] * 100:5.1f}%  (z = {r['z']:+.2f}, "
                     f"n={r['n_flagged']})")

    if s.get("by_score"):
        L += ["", "By how loud the flag was:"]
        for k, v in s["by_score"].items():
            L.append(f"  {k:<8} breakout {v['breakout'] * 100:5.1f}%   "
                     f"move {v['abs_move_z']:.2f}   (n={v['n']})")

    if s.get("by_oi_confirmation"):
        r = s["by_oi_confirmation"]
        L += ["", "Split by whether next-morning open interest confirmed the "
              "trade:",
              f"  confirmed {r['flagged'] * 100:5.1f}%   unconfirmed "
              f"{r['control'] * 100:5.1f}%   (z = {r['z']:+.2f})"]

    p = s.get("power") or {}
    L += ["", "-" * 68]
    if not s.get("enough"):
        L += [f"NOT ENOUGH YET. {p.get('have', 0)} flagged observations; an "
              f"eight-point effect needs about {p.get('need_for_effect')}.",
              "Everything above is printed so the pipeline can be checked, not",
              "so the numbers can be believed."]
    else:
        z = (b or {}).get("z", 0)
        if abs(z) < 2:
            L += ["No effect at the usual bar. The smallest gap this many",
                  f"observations could resolve is about "
                  f"{(p.get('detectable_now') or 0) * 100:.1f} points, so a",
                  "real but smaller effect would still be invisible here.",
                  "",
                  "If this stays flat as the sample grows, the honest move is",
                  "to stop running the scan rather than to loosen the filters",
                  "until something looks significant."]
        else:
            L += [f"Flagged names behave differently (z = {z:+.2f}). That is a",
                  "difference in dispersion, not a trade - the band says how",
                  "far, never which way."]
    return "\n".join(L)


def render_html(s: dict) -> str:
    from dashboard import CSS
    body = report(s).replace("&", "&amp;").replace("<", "&lt;")
    when = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>Flow, graded</title>"
            f"<style>{CSS}pre{{white-space:pre-wrap;font:12.5px/1.6 "
            "ui-monospace,SFMono-Regular,Menlo,monospace}}</style></head><body>"
            '<div class="wrap"><h1>Options flow, graded forward</h1>'
            f'<p class="sub">Updated {when}. Every flag here was archived on '
            "the day it was raised and graded after the week closed.</p>"
            f'<div class="note"><pre>{body}</pre></div>'
            '<footer><a href="./flow.html">Back to today\'s flags</a> &middot; '
            '<a href="./">Ranges</a></footer></div></body></html>')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true")
    p.add_argument("--horizon", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(message)s")

    df = build(args.horizon)
    s = summarise(df)
    print(report(s))

    if args.write:
        (config.DOCS / "flow_results.html").write_text(render_html(s))
        (config.DATA / "flow_results.json").write_text(
            json.dumps(s, indent=2, default=str))
        print(f"\nwrote {config.DOCS / 'flow_results.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
