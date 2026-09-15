"""The options page.

One question per row, asked in words: how far does the model think this moves,
what are options charging for the same stretch of time, and which way — if any —
do the three directional inputs point.

The design rules, learned from the page this replaces:

* **One number per question.** The old page showed a watch score, a percentile
  inside a range published last week, and two dollar ranges, and a reader could
  not tell which of them was the answer to anything.
* **Say what it means in a sentence.** Not "ratio 1.41" but "options are
  charging about 40% more than the model expects".
* **Show how much each input is worth.** The lean is assembled from three
  things with very different amounts of evidence behind them, and the page
  labels which is which rather than averaging them into a number that hides it.
* **Never imply a direction from the top half.** The range is about distance.
  The lean is separate, below, and visually quieter.
"""
from __future__ import annotations

import html

CSS = """
:root{
  --bg:#f6f7f9; --card:#fff; --line:#e4e7ec; --ink:#16191f; --dim:#606a78;
  --faint:#98a1ae; --model:#3b7dd8; --implied:#c2882b;
  --rich:#b4463c; --cheap:#2f8a5b; --fair:#6b7280;
  --up:#2f8a5b; --down:#b4463c;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#0f1115; --card:#171a21; --line:#252a34; --ink:#e8eaed;
         --dim:#9aa2ad; --faint:#6b7280; --model:#4ea1ff; --implied:#d29922;
         --rich:#f85149; --cheap:#3fb950; --fair:#8b95a3;
         --up:#3fb950; --down:#f85149; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  padding:20px 14px 60px}
.wrap{max-width:700px;margin:0 auto}
h1{font-size:21px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13.5px;margin:0 0 14px}
.strip{display:flex;flex-wrap:wrap;gap:18px;background:var(--card);
  border:1px solid var(--line);border-radius:10px;padding:11px 15px;
  margin-bottom:16px}
.strip div{font-size:12px;color:var(--dim)}
.strip b{display:block;font-size:17px;color:var(--ink);
  font-variant-numeric:tabular-nums}
input[type=search]{width:100%;padding:11px 13px;font-size:16px;
  background:var(--card);color:var(--ink);border:1px solid var(--line);
  border-radius:9px;margin-bottom:14px}
.legend{display:flex;gap:18px;font-size:12.5px;color:var(--dim);
  margin:0 0 18px;flex-wrap:wrap}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;
  margin-right:5px;vertical-align:-1px}

.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:16px 18px 18px;margin-bottom:14px}
.head{display:flex;align-items:baseline;gap:10px}
.tk{font-size:20px;font-weight:700;letter-spacing:-.01em}
.px{font-size:16px;color:var(--dim);font-variant-numeric:tabular-nums}
.shaky{display:inline-block;background:#6b7280;color:#fff;font-size:9.5px;
  font-weight:700;border-radius:3px;padding:1px 4px;margin-left:4px;
  letter-spacing:.03em;vertical-align:1px}
.earn{margin-left:auto;background:#fdf0d5;color:#8a6414;font-size:11px;
  font-weight:700;border-radius:20px;padding:3px 9px}

.row{padding:14px 0 4px;border-top:1px solid var(--line);margin-top:12px}
.rowhead{display:flex;align-items:baseline;gap:8px;margin-bottom:10px}
.per{font-size:13px;font-weight:700;text-transform:uppercase;
  letter-spacing:.06em;color:var(--faint)}
.verdict{margin-left:auto;font-size:13px;font-weight:700;border-radius:20px;
  padding:3px 10px}
.v-expensive{background:rgba(180,70,60,.12);color:var(--rich)}
.v-cheap{background:rgba(47,138,91,.13);color:var(--cheap)}
.v-fair{background:rgba(107,114,128,.13);color:var(--fair)}
.v-unknown{background:rgba(107,114,128,.10);color:var(--faint)}

.bar{display:flex;align-items:center;gap:10px;margin:7px 0}
.lab{width:74px;font-size:12.5px;color:var(--dim);flex:none}
.track{flex:1;height:14px;background:rgba(128,138,152,.16);border-radius:7px;
  position:relative;overflow:hidden}
.fill{position:absolute;top:0;bottom:0;left:0;border-radius:7px}
.f-model{background:var(--model)}
.f-implied{background:var(--implied)}
.amt{width:62px;text-align:right;font-size:13.5px;font-weight:700;
  font-variant-numeric:tabular-nums;flex:none}
.says{font-size:14.5px;line-height:1.55;margin-top:8px}
.says b{font-variant-numeric:tabular-nums}
.range{color:var(--dim);font-size:13px;margin-top:4px;
  font-variant-numeric:tabular-nums}

.lean{margin-top:12px;background:rgba(128,138,152,.07);
  border:1px solid var(--line);border-radius:9px;padding:10px 12px}
.leanhead{display:flex;align-items:center;gap:8px;font-size:14.5px;
  font-weight:700}
.arrow{font-size:15px}
.up{color:var(--up)} .down{color:var(--down)} .flat{color:var(--faint)}
.strength{margin-left:auto;font-size:11.5px;font-weight:700;
  letter-spacing:.04em;text-transform:uppercase;color:var(--faint)}
.drv{display:flex;gap:8px;align-items:baseline;padding:3px 0;font-size:13px;
  color:var(--dim)}
.drv .n{width:92px;flex:none;color:var(--faint)}
.drv .v{flex:1}
.tag{font-size:10.5px;font-weight:700;border-radius:4px;padding:1px 5px;
  letter-spacing:.03em;white-space:nowrap}
.t-measured{background:rgba(59,125,216,.15);color:var(--model)}
.t-ungraded{background:rgba(128,138,152,.18);color:var(--faint)}
.note{background:var(--card);border:1px solid var(--line);
  border-left:3px solid var(--implied);border-radius:8px;padding:13px 15px;
  font-size:13px;color:var(--dim);margin-bottom:18px;line-height:1.6}
.note b{color:var(--ink)}
footer{color:var(--faint);font-size:12.5px;margin-top:26px;line-height:1.6}
footer a{color:var(--model)}
"""

JS = """
const box = document.getElementById('f');
if (box) box.addEventListener('input', () => {
  const q = box.value.trim().toUpperCase();
  document.querySelectorAll('.card').forEach(c => {
    c.hidden = q && !c.dataset.t.startsWith(q);
  });
});
"""

PERIOD = {"1 week": "Next week", "1 month": "Next month"}


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _sentence(h: dict) -> str:
    m, i = h.get("model_move_pct"), h.get("implied_move_pct")
    if not m:
        return "No range published for this period."
    if not i:
        return (f"Model expects about <b>{m:.1f}%</b> either way. No usable "
                f"option quotes for this period, so there is nothing to "
                f"compare it against.")
    gap = h.get("gap_pct") or 0
    if h["verdict"] == "fair":
        return (f"Model <b>{m:.1f}%</b>, options <b>{i:.1f}%</b>. Close enough "
                f"that there is nothing to act on.")
    more = "more" if gap > 0 else "less"
    return (f"Model expects about <b>{m:.1f}%</b> either way. Options are "
            f"charging <b>{i:.1f}%</b> &mdash; roughly <b>{abs(gap):.0f}% "
            f"{more}</b> than the model expects.")


def _bars(h: dict) -> str:
    m, i = h.get("model_move_pct"), h.get("implied_move_pct")
    if not m and not i:
        return ""
    top = max(m or 0, i or 0) or 1.0
    rows = []
    for lab, val, cls in (("model", m, "f-model"), ("options", i, "f-implied")):
        if val is None:
            rows.append(f'<div class="bar"><span class="lab">{lab}</span>'
                        f'<span class="track"></span>'
                        f'<span class="amt">&mdash;</span></div>')
            continue
        pct = max(4.0, min(100.0, 100.0 * val / top))
        rows.append(
            f'<div class="bar"><span class="lab">{lab}</span>'
            f'<span class="track"><span class="fill {cls}" '
            f'style="width:{pct:.0f}%"></span></span>'
            f'<span class="amt">&plusmn;{val:.1f}%</span></div>')
    return "".join(rows)


def _lean(lean: dict) -> str:
    if not lean or not lean.get("drivers"):
        return ""
    d = lean.get("direction", "flat")
    glyph = {"up": "&#9650;", "down": "&#9660;"}.get(d, "&#9679;")
    word = {"up": "Leans up", "down": "Leans down"}.get(d, "No lean")
    rows = "".join(
        f'<div class="drv"><span class="n">{_e(x["name"])}</span>'
        f'<span class="v">{_e(x["detail"])}'
        + (f'<span class="tag t-{_e(x.get("evidence"))}">'
           f'{"MEASURED: 51.5%" if x.get("evidence") == "measured" else "NOT YET GRADED"}'
           f'</span>' if x.get("evidence") else "")
        + '</span></div>'
        for x in lean["drivers"])
    return (f'<div class="lean"><div class="leanhead">'
            f'<span class="arrow {_e(d)}">{glyph}</span>{word}'
            f'<span class="strength">{_e(lean.get("strength"))}</span></div>'
            f'{rows}</div>')


def _row(label: str, h: dict, lean: dict | None) -> str:
    v = h.get("verdict", "unknown")
    chip = {"expensive": "Options look expensive", "cheap": "Options look cheap",
            "fair": "Fairly priced"}.get(v, "Not priced")
    rng = ""
    if h.get("low") and h.get("high"):
        rng = (f'<div class="range">Half the time it finishes between '
               f'${h["low"]:,.2f} and ${h["high"]:,.2f}</div>')
    return (f'<div class="row"><div class="rowhead">'
            f'<span class="per">{_e(PERIOD.get(label, label))}</span>'
            f'<span class="verdict v-{_e(v)}">{chip}</span></div>'
            f'{_bars(h)}<div class="says">{_sentence(h)}</div>{rng}'
            f'{_lean(lean) if lean else ""}</div>')


def _card(rec: dict) -> str:
    earn = any((h.get("earnings_inside") == 1)
               for h in (rec.get("horizons") or {}).values())
    badge = ('<span class="earn">EARNINGS INSIDE</span>' if earn else "")
    # A wide or stale bid/ask makes a fairly-priced option read cheap, which is
    # the single most likely way this page misleads. Those names still appear -
    # deleting them would hide coverage - but below every clean reading, and
    # saying so on the card.
    if rec.get("quotes_suspect"):
        badge += '<span class="shaky">THIN QUOTES</span>' 
    labels = sorted((rec.get("horizons") or {}).items(),
                    key=lambda kv: kv[1].get("days") or 0)
    # The lean sits under the shortest horizon only. Repeating it under every
    # period would read as independent confirmation of itself.
    rows = "".join(_row(k, h, rec.get("lean") if n == 0 else None)
                   for n, (k, h) in enumerate(labels))
    return (f'<div class="card" data-t="{_e(rec["ticker"])}">'
            f'<div class="head"><span class="tk">{_e(rec["ticker"])}</span>'
            f'<span class="px">${rec.get("close"):,.2f}</span>{badge}</div>'
            f'{rows}</div>')


NOTE = """
<div class="note">
<b>Cheapest first.</b> The page is ordered by how far below the model's
expectation the market is pricing a move, on each name's single best horizon.
The top of the page is where options cost least relative to what the model
thinks is coming; scroll to the bottom for the ones priced richest.
<br><br>
<b>Cheap is not the same as a good trade.</b> An option is often cheap because
the market knows something the model does not &mdash; a deal closing, a
catalyst passing, a stock about to go quiet. The gap is the start of the
question.
<br><br>
<b>What the two bars are.</b> The model's published range is the middle half of
outcomes; an option's implied move is a one-standard-deviation move. Those are
different units, so the percentages here convert the model's range onto the
option's footing &mdash; otherwise every stock on the page would look expensive,
which would be a unit error rather than a signal.
<br><br>
<b>A gap is a question, not an answer.</b> The market prices things the model
cannot see. Whether these gaps predict anything is being recorded daily and has
not yet been graded.
</div>
"""

FOOTER = """
<b>How to read the lean.</b> Three inputs, and they are not equal. The model
tilt is the only one with a number behind it &mdash; measured on 543,000
out-of-sample windows, it calls direction right 51.5% of the time, which is real
and small. Options flow is plausible and completely ungraded. Shape says which
tail is longer.
<br><br>
<b>When the lean matters.</b> If you are trading the size of the move, only the
top half of each row applies &mdash; a straddle does not care which way. The
lean bears only on choosing one side over the other.
<br><br>
Every input is graded forward, and anything that measures as nothing gets
removed rather than quietly kept.
"""


def render(payload: dict, min_gap: float = 0.0) -> str:
    recs = payload.get("tickers") or []
    if min_gap > 0:
        recs = [r for r in recs
                if any(abs((h.get("ratio") or 1.0) - 1.0) >= min_gap
                       for h in r["horizons"].values())]
    cov = payload.get("coverage") or {}
    strip = "".join(
        f"<div>{k}<b>{_e(v)}</b></div>" for k, v in (
            ("names priced", cov.get("priced")),
            ("reading cheap", cov.get("cheap")),
            ("no option chain", cov.get("no_chain")),
            ("forecast today", cov.get("forecast")),
        ) if v is not None)

    body = ("".join(_card(r) for r in recs) if recs else
            '<div class="note">Nothing to show. Either no forecasts were '
            'published today, or no option chains came back.</div>')

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Options view</title>"
        f"<style>{CSS}</style></head><body><div class=\"wrap\">"
        "<h1>Options view</h1>"
        f'<p class="sub">How far it should move, what options cost, and which '
        f'way things lean &middot; {_e(payload.get("asof"))}</p>'
        f'<div class="strip">{strip}</div>{NOTE}'
        '<div class="legend">'
        '<span><span class="dot" style="background:var(--model)"></span>'
        'what the model expects</span>'
        '<span><span class="dot" style="background:var(--implied)"></span>'
        'what options are charging</span></div>'
        '<input type="search" id="f" placeholder="Filter by ticker">'
        f"{body}"
        f'<footer>{FOOTER}<br><br><a href="./">Ranges</a> &middot; '
        '<a href="./results.html">How the ranges have scored</a> &middot; '
        '<a href="./flow.html">Options flow</a></footer>'
        f"</div><script>{JS}</script></body></html>")
