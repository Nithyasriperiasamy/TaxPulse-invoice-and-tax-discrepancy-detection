"""TaxPulse: invoice discrepancy intelligence (Streamlit app)."""

import hashlib
import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from taxpulse_pipeline import DEFAULT_MODEL_PATH, analyze_invoice, load_bundle
from taxpulse_rules import RULE_CATEGORIES

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_invoices"

st.set_page_config(
    page_title="TaxPulse | Invoice discrepancy checks",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1 { font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0; }

.tp-sub { opacity: .7; margin: 0 0 1.4rem 0; font-size: 1.02rem; }

.tp-verdict {
    border-left: 6px solid var(--tp-color);
    background: var(--tp-bg);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: .4rem 0 1.1rem 0;
}
.tp-verdict .tp-head { font-size: 1.45rem; font-weight: 600; color: var(--tp-color); }
.tp-verdict .tp-detail { margin-top: .2rem; opacity: .9; }
.tp-review    { --tp-color:#d92d20; --tp-bg:rgba(217,45,32,.10); }
.tp-attention { --tp-color:#dc6803; --tp-bg:rgba(220,104,3,.11); }
.tp-clear     { --tp-color:#12a05c; --tp-bg:rgba(18,160,92,.11); }
.tp-unclear   { --tp-color:#667085; --tp-bg:rgba(102,112,133,.13); }

.tp-card {
    border: 1px solid rgba(128,128,128,.28);
    border-left: 5px solid var(--tp-color);
    border-radius: 6px;
    padding: .8rem 1rem;
    margin-bottom: .7rem;
}
.tp-card.fail { --tp-color:#d92d20; }
.tp-card.warn { --tp-color:#dc6803; }
.tp-card .tp-title { font-weight: 600; }
.tp-card .tp-tag {
    display:inline-block; font-size:.72rem; font-weight:600; padding:.05rem .5rem;
    border-radius: 99px; margin-right:.5rem; color:#fff; background: var(--tp-color);
    vertical-align: middle;
}
.tp-card .tp-expl { margin: .35rem 0 .5rem 0; opacity:.9; }
.tp-compare { display:flex; gap:2.2rem; flex-wrap:wrap; font-size:.92rem; }
.tp-compare span { display:block; opacity:.6; font-size:.78rem; }
.tp-compare b { font-variant-numeric: tabular-nums; }

.tp-status { font-size:.9rem; line-height:1.9; }
.tp-dot { display:inline-block; width:.6rem; height:.6rem; border-radius:50%;
          margin-right:.5rem; background:#12a05c; }
.tp-dot.off { background:#98a2b3; }
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.25); border-radius: 6px; padding: .6rem .9rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner=False)
def get_bundle(path):
    return load_bundle(path)


def esc(value):
    return html.escape("" if value is None else str(value))


def file_digest(data):
    return hashlib.sha256(data).hexdigest()


def findings_to_frame(findings):
    return pd.DataFrame(
        [
            {
                "Status": f["status"],
                "Category": f["title"],
                "Check": f["check"],
                "Expected": f["expected"],
                "Found": f["actual"],
                "Difference": f["difference"],
                "Explanation": f["explanation"],
            }
            for f in findings
        ]
    )


def render_finding_card(f):
    tag = "Failed" if f["status"] == "fail" else "Confirm"
    compare = ""
    if f["expected"] or f["actual"]:
        parts = [("Expected", f["expected"]), ("Found on invoice", f["actual"])]
        if f["difference"]:
            parts.append(("Difference", f["difference"]))
        compare = '<div class="tp-compare">' + "".join(
            f"<div><span>{esc(label)}</span><b>{esc(value)}</b></div>"
            for label, value in parts
            if value
        ) + "</div>"
    st.markdown(
        f"""<div class="tp-card {f['status']}">
        <div class="tp-title"><span class="tp-tag">{tag}</span>{esc(f['title'])}: {esc(f['check'])}</div>
        <div class="tp-expl">{esc(f['explanation'])}</div>{compare}</div>""",
        unsafe_allow_html=True,
    )


def run_analysis(data, name, bundle, bundle_error):
    """Run the pipeline with visible progress and return the analysis."""
    with st.status("Analysing invoice…", expanded=True) as status:
        st.write("Reading the PDF and extracting fields")
        analysis = analyze_invoice(
            data, file_name=name, bundle=bundle, bundle_error=bundle_error
        )
        st.write("Recalculating amounts and checking completeness")
        st.write(
            "Scoring risk signals"
            if analysis["ml_results"] is not None
            else "Risk-signal models unavailable, continuing with rule checks"
        )
        st.write("Building the PDF report")
        status.update(label="Analysis complete", state="complete", expanded=False)
    return analysis


# ------------------------------------------------------------------ sidebar
bundle, bundle_error = get_bundle(str(DEFAULT_MODEL_PATH))

with st.sidebar:
    st.markdown("### TaxPulse")
    st.caption("Invoice discrepancy checks for finance and audit teams.")

    st.markdown("**How it works**")
    st.markdown(
        "1. Upload a text-based invoice PDF\n"
        "2. TaxPulse extracts the fields and recalculates the amounts\n"
        "3. Review the findings and download the report"
    )

    st.markdown("**System status**")
    ml_on = bundle is not None
    st.markdown(
        '<div class="tp-status">'
        '<div><span class="tp-dot"></span>Rule checks active</div>'
        f'<div><span class="tp-dot {"" if ml_on else "off"}"></span>'
        f'Risk-signal models {"loaded" if ml_on else "unavailable"}</div></div>',
        unsafe_allow_html=True,
    )
    if bundle_error:
        st.caption(bundle_error + " Run `python train_models.py` to create one.")

    samples = sorted(SAMPLE_DIR.glob("*.pdf")) if SAMPLE_DIR.exists() else []
    if samples:
        st.markdown("**Try a sample**")
        for sample in samples:
            st.download_button(
                f"Download {sample.stem.replace('_', ' ')}",
                data=sample.read_bytes(),
                file_name=sample.name,
                mime="application/pdf",
                key=f"sample_{sample.stem}",
            )

    if st.session_state.get("analysis") and st.button("Clear results"):
        for key in ("analysis", "digest", "error"):
            st.session_state.pop(key, None)
        st.rerun()

# --------------------------------------------------------------------- main
st.title("TaxPulse")
st.markdown(
    '<p class="tp-sub">Upload an invoice to see what does not add up, '
    "and download a report you can attach to the review.</p>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Invoice PDF",
    type=["pdf"],
    help="Text-based PDFs only. Scanned images need OCR first.",
    label_visibility="collapsed",
)

if uploaded is not None:
    data = uploaded.getvalue()
    digest = file_digest(data)
    if st.session_state.get("digest") != digest:
        st.session_state.pop("error", None)
        try:
            st.session_state["analysis"] = run_analysis(
                data, uploaded.name, bundle, bundle_error
            )
            st.session_state["digest"] = digest
        except Exception as error:
            st.session_state.pop("analysis", None)
            st.session_state["digest"] = digest
            st.session_state["error"] = str(error)

if st.session_state.get("error"):
    st.error(st.session_state["error"])

analysis = st.session_state.get("analysis")

if analysis is None and not st.session_state.get("error"):
    st.info("Drop an invoice PDF above to begin. The sidebar has sample files if you want to try it first.")

if analysis is not None:
    record = analysis["record"]
    findings = analysis["rule_findings"]
    level, headline, detail = analysis["verdict"]

    st.markdown(
        f"""<div class="tp-verdict tp-{level}">
        <div class="tp-head">{esc(headline)}</div>
        <div class="tp-detail">{esc(detail)}</div></div>""",
        unsafe_allow_html=True,
    )

    counts = {k: sum(f["status"] == k for f in findings) for k in ("fail", "warn", "pass", "na")}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Failed checks", counts["fail"])
    m2.metric("To confirm", counts["warn"])
    m3.metric("Passed", counts["pass"])
    m4.metric("Not evaluable", counts["na"])

    st.write("")
    d1, d2, d3, _ = st.columns([2, 1.3, 1.3, 2])
    stem = Path(analysis["file_name"]).stem
    d1.download_button(
        "Download PDF report",
        data=analysis["report_pdf"],
        file_name=f"{stem}_taxpulse_report.pdf",
        mime="application/pdf",
        type="primary",
        key="dl_pdf",
    )
    d2.download_button(
        "Findings (CSV)",
        data=findings_to_frame(findings).to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_findings.csv",
        mime="text/csv",
        key="dl_csv",
    )
    d3.download_button(
        "Extracted data (JSON)",
        data=json.dumps(record, indent=2, default=str).encode("utf-8"),
        file_name=f"{stem}_extracted.json",
        mime="application/json",
        key="dl_json",
    )

    tab_findings, tab_data, tab_ml, tab_text = st.tabs(
        ["Findings", "Extracted data", "Risk signals", "Source text"]
    )

    # ---- Findings
    with tab_findings:
        issues = [f for f in findings if f["status"] in ("fail", "warn")]
        if issues:
            for f in sorted(issues, key=lambda x: x["status"] != "fail"):
                render_finding_card(f)
        else:
            st.success("No failed checks and nothing to confirm.")

        passed = [f for f in findings if f["status"] == "pass"]
        skipped = [f for f in findings if f["status"] == "na"]
        if passed:
            with st.expander(f"Passed checks ({len(passed)})"):
                for f in passed:
                    st.markdown(f"**{f['title']}**: {f['check']}. {f['explanation']}")
        if skipped:
            with st.expander(f"Could not be evaluated ({len(skipped)})"):
                for f in skipped:
                    st.markdown(f"**{f['title']}**: {f['explanation']}")

    # ---- Extracted data
    with tab_data:
        if analysis["warnings"]:
            for w in analysis["warnings"]:
                st.warning(w)

        cur = record.get("document_currency") or ""

        def money(v):
            return f"{cur} {v:,.2f}".strip() if v is not None else "Not found"

        left, right = st.columns(2)
        with left:
            st.markdown("**Supplier**")
            st.table(
                pd.DataFrame(
                    {
                        "Field": ["Name", "Address", "Phone", "Email", "Registration"],
                        "Value": [
                            record.get("supplier_name") or "Not found",
                            record.get("supplier_address") or "Not found",
                            record.get("supplier_phone") or "Not found",
                            record.get("supplier_email") or "Not found",
                            record.get("supplier_company_registrations") or "Not found",
                        ],
                    }
                ).set_index("Field")
            )
        with right:
            st.markdown("**Invoice**")
            st.table(
                pd.DataFrame(
                    {
                        "Field": [
                            "Invoice number", "PO / reference", "Invoice date",
                            "Due date", "Subtotal", "Tax", "Total",
                        ],
                        "Value": [
                            record.get("invoice_number") or "Not found",
                            record.get("po_number") or record.get("reference_numbers") or "Not found",
                            record.get("document_date") or "Not found",
                            record.get("due_date") or "Not found",
                            money(record.get("document_total_net")),
                            money(record.get("document_total_tax")),
                            money(record.get("document_total_amount")),
                        ],
                    }
                ).set_index("Field")
            )

        st.markdown("**Line items**")
        items = record.get("line_items") or []
        if items:
            frame = pd.DataFrame(items).rename(
                columns={
                    "description": "Description", "quantity": "Qty",
                    "unit_price": "Unit price", "amount": "Amount",
                    "tax_rate": "Tax rate", "tax_amount": "Tax",
                }
            )
            st.dataframe(frame, hide_index=True)
        else:
            st.caption("No line-item table was recognised in this PDF.")

    # ---- Risk signals
    with tab_ml:
        results = analysis.get("ml_results")
        if results:
            st.caption(
                "Naive Bayes models estimate how closely this invoice resembles "
                "invoices with each kind of discrepancy. They are secondary to the "
                "exact checks and were trained on synthetic data, so treat a signal "
                "as a prompt to look, not as a conclusion."
            )
            frame = pd.DataFrame(
                {
                    "Category": [r["title"] for r in results],
                    "Probability": [
                        None if r["probability"] is None else round(r["probability"] * 100, 1)
                        for r in results
                    ],
                    "Threshold": [round(r["threshold"] * 100, 1) for r in results],
                    "Signal": [
                        "Not evaluable" if not r.get("evaluable", True)
                        else ("Flagged" if r["flagged"] else "Not flagged")
                        for r in results
                    ],
                }
            )
            st.dataframe(
                frame,
                hide_index=True,
                column_config={
                    "Probability": st.column_config.ProgressColumn(
                        "Probability", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "Threshold": st.column_config.NumberColumn("Threshold", format="%.1f%%"),
                },
            )
            for r in results:
                if r["flagged"] and r.get("evidence"):
                    with st.expander(f"Evidence: {r['title']}"):
                        st.dataframe(
                            pd.DataFrame(r["evidence"])[["label", "value", "direction"]].rename(
                                columns={"label": "Feature", "value": "Value", "direction": "Leans toward"}
                            ),
                            hide_index=True,
                        )
        else:
            st.info(
                "The risk-signal models are not available, so only the exact checks "
                "were run. " + (analysis.get("ml_error") or "")
            )

    # ---- Source text
    with tab_text:
        st.caption("Text read from the PDF. Compare it with the extracted fields if something looks wrong.")
        st.text_area(
            "PDF text",
            analysis["extracted_text"],
            height=320,
            disabled=True,
            label_visibility="collapsed",
        )
