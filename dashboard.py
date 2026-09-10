"""The forecast page.

Five hundred rows, so this is a filterable table rather than a stack of cards.
The design rules are the ones the football pages arrived at the hard way:

* **Say the sentence, not the symbol.** "Half the time between $184 and $203",
  not a signed number the reader has to orient themselves against.
* **State the record on the page.** The calibration numbers sit at the top,
  including the ones that are unflattering, because a range with no track
  record is decoration.
* **Never imply a direction.** The midpoint is deliberately not called a
  target or a forecast price. The model does not predict direction and the
  page must not look as though it does.
* **Missing means missing.** A horizon with no trained model is absent, not
  filled in.
"""
from __future__ import annotations

import html
import json

CSS = """
:root{
  --bg:#0f1115; --card:#171a21; --line:#252a34; --ink:#e8eaed; --dim:#9aa2ad;
  --faint:#6b7280; --accent:#4ea1ff; --warn:#d29922; --band:#2b4a6b;
}
@media (prefers-color-scheme: light){
  :root{ --bg:#f6f7f9; --card:#fff; --line:#e3e6ea; --ink:#14171c;
         --dim:#5b6472; --faint:#8b95a3; --band:#cfe2f5; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:26px 16px 64px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px;margin:0 0 20px}
.note{background:var(--card);border:1px solid var(--line);
  border-left:3px solid var(--warn);border-radius:8px;padding:14px 16px;
  margin-bottom:18px;font-size:13px;color:var(--dim)}
.note b{color:var(--ink)}
.strip{display:flex;flex-wrap:wrap;gap:20px;background:var(--card);
  border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-bottom:18px}
.strip div{font-size:12px;color:var(--dim)}
.strip b{display:block;font-size:17px;color:var(--ink);font-variant-numeric:tabular-nums}
input[type=search]{width:100%;padding:10px 12px;font-size:15px;
  background:var(--card);color:var(--ink);border:1px solid var(--line);
  border-radius:8px;margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;color:var(--faint);font-weight:600;font-size:11px;
  text-transform:uppercase;letter-spacing:.05em;padding:8px 10px;
  border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg)}
td{padding:9px 10px;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums;vertical-align:top}
tr:hover td{background:var(--card)}
.tk{font-weight:600}
.px{color:var(--dim)}
.rng{white-space:nowrap}
.pct{color:var(--faint);font-size:11.5px;display:block}
.earn{display:inline-block;background:var(--warn);color:#1a1200;font-size:9.5px;
  font-weight:700;border-radius:3px;padding:1px 4px;margin-left:4px;
  letter-spacing:.03em;vertical-align:1px}
.scroll{overflow-x:auto}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:var(--ink)}
th.sortable::after{content:" \2195";opacity:.35;font-size:10px}
.watch{font-weight:600;font-variant-numeric:tabular-nums}
.why{display:block;color:var(--faint);font-size:11px;font-weight:400}
.edge-hi{color:#3fb950}
.edge-lo{color:#f85149}
.pos{white-space:nowrap}
.pos small{color:var(--faint);display:block;font-size:11px}
.lean{font-weight:700;margin-right:4px}
.lean-up{color:#3fb950}
.lean-down{color:#f85149}
.lean-flat{color:var(--faint)}
footer{color:var(--faint);font-size:12px;margin-top:30px;text-align:center}
.cal{font-size:12.5px;color:var(--dim);margin-top:6px}
.cal table{font-size:12px;margin-top:6px}
"""

JS = """
const box = document.getElementById('f');
if (box) box.addEventListener('input', () => {
  const q = box.value.trim().toUpperCase();
  document.querySelectorAll('tbody tr').forEach(r => {
    r.hidden = q && !r.dataset.t.startsWith(q);
  });
});

// Click a header to sort. Numeric columns carry a data-v on each cell so the
// sort reads the underlying value rather than the formatted text - "$1,204.00"
// and "$98.10" sort the wrong way round as strings.
const tbody = document.querySelector('tbody');
document.querySelectorAll('th.sortable').forEach((th, i) => {
  let desc = true;
  th.addEventListener('click', () => {
    const idx = Array.from(th.parentNode.children).indexOf(th);
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {
      const av = a.children[idx].dataset.v, bv = b.children[idx].dataset.v;
      if (av !== undefined && bv !== undefined) {
        const d = parseFloat(bv) - parseFloat(av);
        return desc ? d : -d;
      }
      const at = a.children[idx].textContent, bt = b.children[idx].textContent;
      return desc ? bt.localeCompare(at) : at.localeCompare(bt);
    });
    rows.forEach(r => tbody.appendChild(r));
    desc = !desc;
  });
});
"""


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _range_cell(h: dict) -> str:
    if not h or h.get("low") is None or h.get("high") is None:
        return '<td class="rng">—</td>'
    flag = ('<span class="earn">EARNINGS</span>'
            if h.get("earnings_inside") == 1 else "")
    pct = ""
    if h.get("pct_low") is not None and h.get("pct_high") is not None:
        pct = (f'<span class="pct">{h["pct_low"]:+.1f}% to '
               f'{h["pct_high"]:+.1f}%</span>')
    width = ((h["pct_high"] - h["pct_low"])
             if h.get("pct_high") is not None and h.get("pct_low") is not None
             else 0)

    lean = h.get("lean")
    rel = h.get("rel_tilt_pct")
    arrow = ""
    if lean == "up":
        arrow = f'<span class="lean lean-up" title="median tilts {rel:+.2f}% above the day\'s typical stock">&#9650;</span>'
    elif lean == "down":
        arrow = f'<span class="lean lean-down" title="median tilts {rel:+.2f}% below the day\'s typical stock">&#9660;</span>'
    elif lean == "flat":
        arrow = '<span class="lean lean-flat" title="in line with the day\'s typical stock">&#9679;</span>'

    return (f'<td class="rng" data-v="{width:.3f}">'
            f'{arrow}${h["low"]:,.2f} – ${h["high"]:,.2f}{flag}{pct}</td>')


def _watch_cell(r: dict) -> str:
    """The ranking number, and one line saying why it is what it is."""
    score = r.get("watch_score") or 0.0
    open_ = (r.get("open_ranges") or [])
    ts = r.get("term_structure") or {}

    if open_:
        top = open_[0]
        pct = top["percentile"]
        cls = "edge-hi" if pct >= 50 else "edge-lo"
        why = (f'{pct:.0f}th pct of the {_e(top["horizon"])} range '
               f'published {_e(top["published"])}')
    elif ts:
        cls = ""
        why = (f'{_e(ts["direction"])} — next {_e(ts["short"])} priced at '
               f'{ts["short_annual_pct"]:.0f}% vs {ts["long_annual_pct"]:.0f}% '
               f'for the {_e(ts["long"])}')
    else:
        cls = ""
        why = ""
    return (f'<td class="watch" data-v="{score}">'
            f'<span class="{cls}">{score:.0f}</span>'
            f'<span class="why">{why}</span></td>')


def _position_cell(r: dict) -> str:
    """Where today sits inside every range still running."""
    open_ = r.get("open_ranges") or []
    if not open_:
        return '<td class="pos" data-v="-1">—</td>'
    lead = open_[0]["percentile"]
    bits = "".join(
        f'<small>{_e(o["horizon"])}: {o["percentile"]:.0f}th '
        f'({o["elapsed_frac"] * 100:.0f}% through)</small>' for o in open_)
    return (f'<td class="pos" data-v="{abs(lead - 50):.2f}">'
            f'{lead:.0f}th{bits}</td>')


def _row(r: dict, labels: list) -> str:
    cells = "".join(_range_cell((r.get("horizons") or {}).get(l)) for l in labels)
    return (f'<tr data-t="{_e(r["ticker"])}">'
            f'<td class="tk">{_e(r["ticker"])}</td>'
            f'{_watch_cell(r)}{_position_cell(r)}'
            f'<td class="px" data-v="{r["close"]}">${r["close"]:,.2f}</td>'
            f'{cells}</tr>')


def _strip(payload: dict) -> str:
    m = (payload.get("model_metrics") or {}).get("horizons") or {}
    cells = [("Tickers", f'{len(payload.get("tickers", []))}'),
             ("As of", payload.get("asof", "—"))]
    for label, block in m.items():
        met = (block or {}).get("metrics") or {}
        if met.get("coverage") is not None:
            cells.append((f"{label} coverage",
                          f'{met["coverage"] * 100:.1f}%'))
    body = "".join(f"<div>{_e(k)}<b>{_e(v)}</b></div>" for k, v in cells)
    return f'<div class="strip">{body}</div>'


def _honesty(payload: dict) -> str:
    m = (payload.get("model_metrics") or {}).get("horizons") or {}
    rows = []
    for label, block in m.items():
        met = (block or {}).get("metrics") or {}
        if not met:
            continue
        rows.append(
            f"<tr><td>{_e(label)}</td>"
            f"<td>{met['coverage'] * 100:.1f}%</td>"
            f"<td>{met['coverage_naive'] * 100:.1f}%</td>"
            f"<td>{met['worst_bucket_error'] * 100:.1f}</td>"
            f"<td>{met['worst_bucket_error_naive'] * 100:.1f}</td>"
            f"<td>{met['pinball_gain_pct']:+.1f}%</td>"
            f"<td>{met['n_independent']:,}</td></tr>")

    table = ""
    if rows:
        table = (
            '<div class="cal"><table>'
            '<tr><th>horizon</th><th>coverage</th><th>naive</th>'
            '<th>worst bucket</th><th>naive</th><th>vs naive</th>'
            '<th>windows</th></tr>' + "".join(rows) + "</table></div>")

    return (
        '<div class="note">'
        '<b>What this is.</b> For each stock, the range its price landed in '
        'half the time in comparable past situations — not a prediction of '
        'which way it moves. The model forecasts how far a stock is likely to '
        'travel, which is forecastable, and says nothing about direction, '
        'which is not.'
        '<br><br><b>How well it has done.</b> Out of sample, on '
        'non-overlapping windows, against the obvious answer of assuming the '
        'next period looks like the last. "Worst bucket" is how far the worst '
        'volatility quartile sat from the 50% target — the number that '
        'separates being right on average from being right everywhere.'
        f'{table}'
        '<br><br><b>The watch column.</b> Every range on this page is drawn '
        'outward from today\'s close, so today sits in the middle of all of '
        'them by construction — asking where a stock is inside its own bracket '
        'has the same answer for all five hundred. Ranking needs a second '
        'reference point, and this uses two.'
        '<br><br>Where available, <b>where today\'s price sits inside a range '
        'published earlier</b> and still running. A stock at the 92nd '
        'percentile of the range forecast for it three weeks ago has moved '
        'further than expected, which is the truest reading of "at the edge '
        'of its bracket". This fills in as the daily runs accumulate.'
        '<br><br>Otherwise, <b>expected turbulence</b>: the near-term band '
        'against the long-term one, both annualised so they compare. A stock '
        'whose next week is priced far wider than its next quarter is one the '
        'model thinks is about to move.'
        '<br><br><b>The arrows.</b> \u25b2 and \u25bc mark whether the '
        'forecast median for that window sits above or below the day\'s '
        'typical stock. Every stock drifts upward over a quarter because the '
        'market does, and that part is identical for all five hundred names '
        'and useless — so the arrow shows only what is left after the crowd is '
        'subtracted. \u25cf means in line with everything else.'
        '<br><br>Treat them as the weakest thing on the page, and here is '
        'exactly how weak. Every training run measures them, and the last one '
        'found the cross-sectional tilt called direction correctly <b>51.3% '
        'of the time</b> at one week — 1.3 points better than a coin, holding '
        'at 51.3% and 51.1% over a month and a quarter.'
        '<br><br>Two things make that smaller than it looks. Those windows '
        'share dates with one another, so the confidence figure printed in the '
        'run is flattering. And 1.3 points is 1.3 points. What the same '
        'measurement found alongside it matters more: <b>without</b> the '
        'cross-sectional subtraction the tilt is actively wrong — 53.6% '
        'against a base rate of 54.5% for simply always saying up. The '
        'subtraction is not a refinement, it is the entire effect. Had the page '
        'shown the raw tilt, it would have pointed the wrong way.'
        '<br><br>So the arrows stay, because a measured 1.3 points is not '
        'zero. They are not a reason to buy anything.'
        '<br><br><b>Nothing here says which way with confidence.</b> The watch '
        'column says a stock already moved or is likely to move; the arrows are '
        'a faint tilt. Green and red mark up and down, not good and bad.'
        '<br><br><b>A calibrated range is not a trading strategy.</b> It says '
        'what is plausible, not what is mispriced.'
        '</div>')


def render(payload: dict, standalone: bool = True) -> str:
    rows = payload.get("tickers") or []
    labels = []
    for r in rows:
        for l in (r.get("horizons") or {}):
            if l not in labels:
                labels.append(l)

    head = "".join(f'<th class="sortable">{_e(l)}</th>' for l in labels)
    body = "".join(_row(r, labels) for r in rows)
    asof = payload.get("asof", "")
    gen = (payload.get("generated_at") or "")[:16].replace("T", " ")

    inner = (
        '<div class="wrap">'
        f'<h1>Stock ranges — {_e(asof)}</h1>'
        f'<p class="sub">Generated {_e(gen)} UTC. Prices are the last close; '
        f'ranges are the middle half of the forecast distribution.</p>'
        f'{_strip(payload)}{_honesty(payload)}'
        '<input type="search" id="f" placeholder="Filter by ticker…" '
        'autocomplete="off">'
        '<div class="scroll"><table><thead><tr>'
        '<th class="sortable">ticker</th>'
        '<th class="sortable">watch</th>'
        '<th class="sortable">in open range</th>'
        '<th class="sortable">last</th>'
        f'{head}</tr></thead><tbody>{body}</tbody></table></div>'
        + ('' if rows else '<p class="sub">No forecasts in this run.</p>')
        + '<footer>Forecasts of range only. Not investment advice.</footer>'
        '</div>')

    if not standalone:
        return inner
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>Stock ranges — {_e(asof)}</title>"
            f"<style>{CSS}</style></head><body>{inner}"
            f"<script>{JS}</script></body></html>")


def render_from_file(path: str) -> str:
    with open(path) as fh:
        return render(json.load(fh))


if __name__ == "__main__":
    import sys
    print(render_from_file(sys.argv[1]))
