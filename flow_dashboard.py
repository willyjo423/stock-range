"""The flow page.

Same rules as the range page, with one addition that matters more here: the
limits of the data are printed on the page rather than in a README nobody
opens. A flow screen built from snapshots looks exactly like a flow screen
built from the tape, and the difference is entirely in what the numbers mean.
So the page says, above the table, that premium is the contract's total for the
interval rather than one trade's size, and that whether it was bought or sold
is a guess unless it is marked firm.

The other rule carried over: nothing here is called bullish or bearish. Call
premium is call premium. Whether someone bought those calls or sold them is not
in the data.
"""
from __future__ import annotations

import html

from dashboard import CSS

EXTRA_CSS = """
.tilt-calls{color:#3fb950;font-weight:600}
.tilt-puts{color:#f85149;font-weight:600}
.tilt-two-sided{color:var(--dim);font-weight:600}
details.c{margin:0}
details.c summary{cursor:pointer;color:var(--accent);font-size:12px;
  list-style:none}
details.c summary::-webkit-details-marker{display:none}
details.c summary::before{content:"\\25b8 ";}
details.c[open] summary::before{content:"\\25be ";}
.ctab{width:100%;margin:6px 0 2px;font-size:12px}
.ctab td{padding:3px 6px;border-bottom:1px dotted var(--line)}
.tag{display:inline-block;font-size:9.5px;font-weight:700;border-radius:3px;
  padding:1px 4px;margin-left:4px;letter-spacing:.03em;vertical-align:1px}
.tag-burst{background:#4ea1ff;color:#04111f}
.tag-ml{background:var(--line);color:var(--dim)}
.tag-stale{background:var(--warn);color:#1a1200}
.prem{font-weight:600}
.empty{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:26px;text-align:center;color:var(--dim)}
"""


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def money(v) -> str:
    """Premium, rounded to something a person reads rather than counts."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}m"
    if v >= 1_000:
        return f"${v / 1_000:.0f}k"
    return f"${v:.0f}"


def _contract_rows(rec: dict) -> str:
    rows = []
    for c in rec.get("contracts") or []:
        tags = ""
        if c.get("burst") is not None and c["burst"] >= 0.6:
            tags += '<span class="tag tag-burst">CONCENTRATED</span>'
        if c.get("stale_mid"):
            tags += '<span class="tag tag-stale">STALE QUOTE</span>'
        rows.append(
            f"<tr><td>{_e(c.get('expiry'))} &middot; "
            f"{_e(c.get('right'))}{_e(c.get('strike'))}</td>"
            f"<td>{_e(c.get('dte'))}d</td>"
            f"<td class=\"prem\">{money(c.get('premium'))}</td>"
            f"<td>{_e(c.get('volume'))} vs {_e(c.get('open_interest'))} OI"
            f" ({_e(c.get('vol_oi'))}x)</td>"
            f"<td>{_e(c.get('side'))}{tags}</td></tr>")
    if not rows:
        return ""
    return ('<details class="c"><summary>contracts</summary>'
            f'<table class="ctab">{"".join(rows)}</table></details>')


def _row(rec: dict) -> str:
    tilt = rec.get("tilt") or "two-sided"
    ml = ('<span class="tag tag-ml">MATCHED LEGS</span>'
          if rec.get("looks_multileg") else "")
    return (
        f'<tr data-t="{_e(rec["ticker"])}">'
        f'<td class="tk">{_e(rec["ticker"])}</td>'
        f'<td class="px" data-v="{_e(rec.get("spot"))}">'
        f'${_e(rec.get("spot"))}</td>'
        f'<td data-v="{_e(rec.get("score"))}" class="watch">'
        f'{_e(rec.get("score"))}</td>'
        f'<td data-v="{_e(rec.get("premium"))}" class="prem">'
        f'{money(rec.get("premium"))}'
        f'<span class="pct">{money(rec.get("call_premium"))} calls / '
        f'{money(rec.get("put_premium"))} puts</span></td>'
        f'<td class="tilt-{_e(tilt)}">{_e(tilt)}{ml}</td>'
        f'<td data-v="{_e(rec.get("nearest_dte"))}">'
        f'{_e(rec.get("nearest_dte"))}d</td>'
        f'<td data-v="{_e(rec.get("max_vol_oi"))}">'
        f'{_e(rec.get("max_vol_oi"))}x</td>'
        f'<td>{_contract_rows(rec)}</td>'
        "</tr>")


def _coverage(cov: dict) -> str:
    if not cov:
        return ""
    bits = [
        ("names scanned", cov.get("with_chain")),
        ("no chain returned", cov.get("failed")),
        ("contracts read", f'{(cov.get("contracts") or 0):,}'),
        ("scans today", cov.get("snapshots_today")),
        ("contracts flagged", cov.get("flagged_contracts")),
    ]
    inner = "".join(f"<div>{_e(k)}<b>{_e(v)}</b></div>" for k, v in bits
                    if v is not None)
    return f'<div class="strip">{inner}</div>'


CAVEAT = """
<div class="note">
<b>What these numbers are, and what they are not.</b>
Free data gives a delayed <i>snapshot</i> of the option chain, not the tape.
That means the premium shown is everything that traded in a contract during the
interval &mdash; however many hands it took &mdash; and not one trade's size.
A single large order and four hundred small ones arrive here as the same
figure. The one thing that separates them is timing: when a contract's whole
day of volume lands inside one scan interval it is marked
<span class="tag tag-burst">CONCENTRATED</span>, which is as close to
&ldquo;one order did that&rdquo; as this can get.
<br><br>
<b>Open interest is the strong filter.</b> It settles overnight and does not
move during the session, so today's volume against it is a genuine
before-and-after. A ratio above 1&times; means more contracts changed hands
today than existed yesterday.
<br><br>
<b>Bought or sold is a guess.</b> Whether premium was paid or collected needs
the quote at the instant of the trade. Where the last print sits inside the
current market is shown when the trade fell inside the interval just measured,
and left as <i>unknown</i> otherwise. Call premium is not the same as bullish:
someone sold every one of those contracts.
</div>
"""


def render(payload: dict) -> str:
    recs = payload.get("tickers") or []
    asof = payload.get("asof", "")
    scans = ", ".join(payload.get("scans") or []) or "one"

    if recs:
        body = (
            '<input type="search" id="f" placeholder="Filter by ticker">'
            '<div class="scroll"><table><thead><tr>'
            "<th>Ticker</th><th>Spot</th><th class=\"sortable\">Score</th>"
            "<th class=\"sortable\">Premium</th><th>Side</th>"
            "<th class=\"sortable\">Nearest</th>"
            "<th class=\"sortable\">Vol/OI</th><th></th>"
            f'</tr></thead><tbody>{"".join(_row(r) for r in recs)}'
            "</tbody></table></div>")
    else:
        body = ('<div class="empty">Nothing cleared every filter on this '
                "scan. That is the ordinary outcome on most days, and a page "
                "that always has something on it is a page with the filters "
                "turned down.</div>")

    from dashboard import JS
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Options flow</title>"
        f"<style>{CSS}{EXTRA_CSS}</style></head><body>"
        '<div class="wrap">'
        "<h1>Short-dated options flow</h1>"
        f'<p class="sub">S&amp;P 500 &middot; expirations inside two weeks '
        f'&middot; at the money &middot; {_e(asof)} &middot; '
        f'scans at {_e(scans)} UTC</p>'
        f"{_coverage(payload.get('coverage') or {})}"
        f"{CAVEAT}"
        f"{body}"
        '<footer><a href="./flow_results.html">Is any of this worth '
        "anything? &mdash; graded forward</a> &middot; "
        '<a href="./">Stock ranges</a></footer>'
        f"</div><script>{JS}</script></body></html>")
