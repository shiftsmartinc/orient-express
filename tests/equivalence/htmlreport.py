"""HTML rendering for the equivalence report.

Pure functions over collected results — used both by the pytest session
(report emitted automatically at the end of a run) and by the standalone
report.py. The output embeds internal photos; treat it as private.
"""

import base64
import html

STYLE = """
body { font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.15rem; margin-top: 2.5rem;
border-bottom: 2px solid #ddd; padding-bottom: .3rem; }
h3 { font-size: 1rem; margin-top: 1.5rem; }
.summary { border-collapse: collapse; margin: 1rem 0; }
.summary td, .summary th { border: 1px solid #ccc; padding: .35rem .7rem; }
.imgpair { display: flex; gap: 1rem; margin: .5rem 0; }
.imgpair figure { margin: 0; }
.imgpair img { max-width: 460px; max-height: 460px; border: 1px solid #bbb; }
figcaption { font-size: .8rem; color: #555; text-align: center; }
table.diff { border-collapse: collapse; font-size: .82rem; margin: .4rem 0 1.4rem; }
table.diff td, table.diff th { border: 1px solid #ddd; padding: .25rem .55rem;
text-align: left; }
tr.fail { background: #ffe3e3; font-weight: 600; }
tr.ok-delta { background: #fff8dc; }
.pass { color: #1a7f37; font-weight: 600; } .fail { color: #c0392b; font-weight: 700; }
.meta { color: #666; font-size: .85rem; }
"""


def img_tag(jpeg_bytes, caption):
    if jpeg_bytes is None:
        return ""
    encoded = base64.b64encode(jpeg_bytes).decode()
    return (
        f'<figure><img src="data:image/jpeg;base64,{encoded}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def diff_table(records):
    rows = []
    for r in records:
        css = "fail" if not r["ok"] else ("ok-delta" if "Δ" in str(r["detail"]) else "")
        status = "FAIL" if not r["ok"] else "ok"
        rows.append(
            f'<tr class="{css}"><td>{html.escape(str(r["field"]))}</td>'
            f"<td>{html.escape(str(r['golden']))}</td>"
            f"<td>{html.escape(str(r['got']))}</td>"
            f"<td>{html.escape(str(r['detail']))}</td>"
            f"<td>{status}</td></tr>"
        )
    return (
        '<table class="diff"><tr><th>field</th><th>golden (before)</th>'
        "<th>current (after)</th><th>detail</th><th>status</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def image_block(title, records, before_jpeg=None, after_jpeg=None, captions=None):
    """One image's section: status header, before/after pair, diff table."""
    failed = any(not r["ok"] for r in records)
    status = (
        '<span class="fail">DIFFERS</span>'
        if failed
        else '<span class="pass">equivalent</span>'
    )
    parts = [f"<h4>{html.escape(title)} — {status}</h4>"]
    if before_jpeg or after_jpeg:
        before_caption, after_caption = captions or (
            "before (golden)",
            "after (current)",
        )
        parts.append(
            '<div class="imgpair">'
            + img_tag(before_jpeg, before_caption)
            + img_tag(after_jpeg, after_caption)
            + "</div>"
        )
    parts.append(diff_table(records))
    return "".join(parts), failed


def build_page(case_sections, current_provenance):
    """case_sections: list of (case_name, html, ok_count, fail_count)."""
    summary_rows = []
    body = []
    for case_name, section_html, ok, fail in case_sections:
        verdict = (
            '<span class="pass">equivalent</span>'
            if fail == 0
            else f'<span class="fail">{fail} differ</span>'
        )
        summary_rows.append(
            f"<tr><td>{html.escape(case_name)}</td><td>{ok}</td>"
            f"<td>{fail}</td><td>{verdict}</td></tr>"
        )
        body.append(section_html)
    return f"""<!doctype html><meta charset="utf-8">
<title>orient-express equivalence report</title>
<style>{STYLE}</style>
<h1>Equivalence report — current vs golden</h1>
<p class="meta">current code: {html.escape(str(current_provenance.get("library_version")))} @
{html.escape(str(current_provenance.get("git_commit")))} on
{html.escape(str(current_provenance.get("platform")))}</p>
<table class="summary"><tr><th>case</th><th>images equivalent</th>
<th>images differing</th><th>verdict</th></tr>{"".join(summary_rows)}</table>
{"".join(body)}
"""
