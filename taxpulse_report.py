
from io import BytesIO
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    PageBreak,
)
from datetime import datetime

from taxpulse_rules import RULE_CATEGORIES


CATEGORY_DETAILS = {
    "tax_amount_mismatch": {
        "title": "Tax Amount Mismatch",
        "why": (
            "This assessment estimates whether the extracted invoice "
            "features resemble patterns associated with tax amount "
            "mismatches in the training data. Verify the reported tax "
            "against the invoice amounts and tax breakdown."
        ),
        "checks": [
            "Recalculate tax using the applicable taxable amount and rates.",
            "Compare the reported tax with the invoice tax breakdown.",
            "Confirm that the correct tax treatment was applied.",
        ],
    },
    "invoice_total_mismatch": {
        "title": "Invoice Total Mismatch",
        "why": (
            "This assessment estimates whether the extracted invoice "
            "features resemble patterns associated with inconsistencies "
            "between reported and calculated invoice totals."
        ),
        "checks": [
            "Recalculate the total from the subtotal, taxes, and adjustments.",
            "Check discounts, rounding, charges, and credits.",
            "Compare the calculated result with the printed invoice total.",
        ],
    },
    "incomplete_supplier_details": {
        "title": "Incomplete Supplier Details",
        "why": (
            "This assessment estimates whether the extracted supplier "
            "information resembles patterns associated with missing "
            "supplier details. It does not verify information against "
            "an external supplier database."
        ),
        "checks": [
            "Verify the supplier's legal name and address.",
            "Check applicable registration and contact details.",
            "Compare supplier information against an approved supplier record.",
        ],
    },
    "line_item_arithmetic": {
        "title": "Line-Item Arithmetic Inconsistency",
        "why": (
            "This assessment estimates whether the extracted invoice "
            "features resemble patterns associated with line-item "
            "calculation errors."
        ),
        "checks": [
            "Recalculate quantity multiplied by unit price.",
            "Check line-level discounts and taxes.",
            "Reconcile the line-item sums with the invoice subtotal.",
        ],
    },
    "tax_breakdown_mismatch": {
        "title": "Tax Breakdown Mismatch",
        "why": (
            "This assessment estimates whether the extracted invoice "
            "features resemble patterns associated with inconsistencies "
            "in taxable amounts, tax rates, or the reported tax breakdown."
        ),
        "checks": [
            "Compare the tax breakdown with the reported tax total.",
            "Verify the taxable bases and tax rates.",
            "Check rounding and tax-component calculations.",
        ],
    },
    "document_reference_inconsistency": {
        "title": "Document Reference Inconsistency",
        "why": (
            "This assessment estimates whether the extracted document "
            "features resemble patterns associated with missing or "
            "inconsistent document references. External consistency "
            "cannot be verified without the relevant source records."
        ),
        "checks": [
            "Verify the invoice or document number.",
            "Check relevant purchase-order or receipt references.",
            "Compare references against source documents when available.",
        ],
    },
}


FEATURE_LABELS = {
    "tax_amount": "Reported tax amount",
    "total_amount": "Reported invoice total",
    "calculated_total": "Calculated invoice total",
    "total_difference": "Difference between reported and calculated totals",
    "absolute_total_difference": "Absolute invoice-total difference",
    "line_net_difference": "Difference between line-item sum and net amount",
    "absolute_line_net_difference": "Absolute line-item/net difference",
    "line_tax_difference": "Difference between line tax and reported tax",
    "absolute_line_tax_difference": "Absolute line-tax difference",
    "line_amount_mismatch_count": "Number of line-amount mismatches",
    "line_amount_absolute_error_sum": "Combined line-amount error",
    "supplier_missing_count": "Number of missing supplier fields",
    "supplier_name_missing": "Supplier name missing indicator",
    "supplier_address_missing": "Supplier address missing indicator",
    "supplier_registration_missing": "Supplier registration missing indicator",
    "supplier_phone_missing": "Supplier phone missing indicator",
    "supplier_email_missing": "Supplier email missing indicator",
    "supplier_website_missing": "Supplier website missing indicator",
    "reference_missing_count": "Number of missing references",
    "all_references_missing": "All document references missing indicator",
    "document_number_missing": "Document number missing indicator",
    "document_number_missing_but_secondary_present":
        "Document number missing despite another reference being present",
    "due_before_document": "Due date occurs before document date",
    "payment_before_document": "Payment date occurs before document date",
    "payment_after_due": "Payment date occurs after due date",
    "payment_before_due": "Payment date occurs before due date",
    "missing_line_amount_count": "Number of line items missing amounts",
    "missing_line_tax_count": "Number of line items missing tax values",
    "missing_quantity_count": "Number of line items missing quantities",
    "missing_unit_price_count": "Number of line items missing unit prices",
    "missing_description_count": "Number of line items missing descriptions",
    "missing_line_tax_rate_count": "Number of line items missing tax rates",
    "currency_conflict_count": "Number of detected currency conflicts",
    "currency_mismatch_flag": "Currency mismatch indicator",
}


def readable_name(name):
    return FEATURE_LABELS.get(
        name,
        name.replace("_", " ").strip().capitalize(),
    )


def get_pipeline_parts(pipeline):
    """
    Return the imputer and final classifier from a pipeline.
    If the supplied model is not a pipeline, return the model itself.
    """
    if hasattr(pipeline, "named_steps"):
        imputer = pipeline.named_steps.get("imputer")
        classifier = pipeline.named_steps.get("model")

        if classifier is None:
            classifier = list(pipeline.named_steps.values())[-1]

        return imputer, classifier

    return None, pipeline


def explain_prediction(pipeline, columns, feature_row, max_features=5):
    """
    Calculate per-feature Gaussian log-likelihood differences.

    Positive contribution:
        The feature value favors Class 1 relative to Class 0.

    Negative contribution:
        The feature value favors Class 0 relative to Class 1.

    Zero contribution:
        The feature value provides no preference between the classes.

    These contributions are not causal explanations and do not represent
    the complete final predicted probability.
    """
    imputer, classifier = get_pipeline_parts(pipeline)

    if imputer is not None:
        values = imputer.transform(feature_row[columns])
    else:
        values = feature_row[columns].to_numpy(dtype=float)

    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[0] != 1:
        return []

    if not hasattr(classifier, "theta_") or not hasattr(classifier, "var_"):
        return []

    classes = np.asarray(classifier.classes_)
    negative_positions = np.flatnonzero(classes == 0)
    positive_positions = np.flatnonzero(classes == 1)

    if len(negative_positions) == 0 or len(positive_positions) == 0:
        return []

    negative_index = int(negative_positions[0])
    positive_index = int(positive_positions[0])

    x = values[0]
    means = np.asarray(classifier.theta_, dtype=float)
    variances = np.asarray(classifier.var_, dtype=float)

    if (
        means.ndim != 2
        or variances.ndim != 2
        or means.shape != variances.shape
        or means.shape[1] != len(columns)
        or negative_index >= means.shape[0]
        or positive_index >= means.shape[0]
    ):
        return []

    # Avoid invalid logarithms or division by zero.
    if (
        np.any(~np.isfinite(variances))
        or np.any(variances <= 0)
    ):
        return []

    log_likelihood_negative = -0.5 * (
        np.log(2 * np.pi * variances[negative_index])
        + (
            (x - means[negative_index]) ** 2
            / variances[negative_index]
        )
    )

    log_likelihood_positive = -0.5 * (
        np.log(2 * np.pi * variances[positive_index])
        + (
            (x - means[positive_index]) ** 2
            / variances[positive_index]
        )
    )

    contributions = (
        log_likelihood_positive - log_likelihood_negative
    )

    valid_indices = [
        index
        for index, contribution in enumerate(contributions)
        if np.isfinite(contribution) and np.isfinite(x[index])
    ]

    valid_indices.sort(
        key=lambda index: abs(contributions[index]),
        reverse=True,
    )

    output = []

    for index in valid_indices[:max_features]:
        contribution = float(contributions[index])

        if contribution > 0:
            direction = "Favors discrepancy"
        elif contribution < 0:
            direction = "Favors no discrepancy"
        else:
            direction = "Neutral"

        output.append({
            "feature": columns[index],
            "label": readable_name(columns[index]),
            "value": float(x[index]),
            "contribution": contribution,
            "direction": direction,
        })

    return output


def predict_invoice(feature_row, model_bundle):
    """
    Predict the discrepancy probability for each configured category.

    The returned probability is the model's estimated probability of
    Class 1. A category is flagged when this probability reaches its
    configured threshold.
    """
    if not isinstance(feature_row, pd.DataFrame):
        raise TypeError("feature_row must be a pandas DataFrame.")

    if len(feature_row) != 1:
        raise ValueError(
            "feature_row must contain exactly one invoice."
        )

    categories = model_bundle.get("categories") or list(model_bundle["models"].keys())
    models = model_bundle["models"]
    model_columns = model_bundle["model_columns"]
    thresholds = model_bundle.get("thresholds", {})

    results = []

    for category in categories:
        if category not in models:
            raise KeyError(f"Missing trained model: {category}")

        if category not in model_columns:
            raise KeyError(
                f"Missing feature-column configuration for {category}."
            )

        columns = model_columns[category]
        pipeline = models[category]

        missing_columns = [
            column
            for column in columns
            if column not in feature_row.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Feature extraction is missing columns for "
                f"{category}: {missing_columns}"
            )

        model_input = feature_row.loc[:, columns]

        # If most inputs are missing the imputer would silently substitute
        # "typical" values and report a clean invoice. Say "not evaluable".
        missing_share = float(model_input.isna().to_numpy().mean())
        if missing_share > 0.5:
            results.append({
                "category": category,
                "title": CATEGORY_DETAILS.get(category, {}).get(
                    "title", readable_name(category)),
                "probability": None,
                "threshold": float(thresholds.get(category, 0.5)),
                "flagged": False,
                "evaluable": False,
                "evidence": [],
            })
            continue

        probabilities = pipeline.predict_proba(model_input)[0]
        _, classifier = get_pipeline_parts(pipeline)

        classes = np.asarray(classifier.classes_)
        positive_positions = np.flatnonzero(classes == 1)

        if len(positive_positions) == 0:
            raise ValueError(
                f"Model for {category} was not trained with class 1."
            )

        positive_probability = float(
            probabilities[int(positive_positions[0])]
        )

        if not np.isfinite(positive_probability):
            raise ValueError(
                f"Model returned an invalid probability for {category}."
            )

        threshold = float(thresholds.get(category, 0.5))

        if not 0 <= threshold <= 1:
            raise ValueError(
                f"Invalid decision threshold for {category}: {threshold}"
            )

        flagged = positive_probability >= threshold

        results.append({
            "category": category,
            "title": CATEGORY_DETAILS.get(
                category, {}
            ).get("title", readable_name(category)),
            "probability": positive_probability,
            "threshold": threshold,
            "flagged": flagged,
            "evaluable": True,
            "evidence": explain_prediction(
                pipeline,
                columns,
                model_input,
            ),
        })

    return results


# =====================================================================
# PDF report
# =====================================================================

NAVY = colors.HexColor("#1F3A5F")
INK = colors.HexColor("#1D2939")
MUTED = colors.HexColor("#667085")
LINE = colors.HexColor("#D0D5DD")
STATUS_STYLE = {
    "fail": ("FAILED", colors.HexColor("#B42318"), colors.HexColor("#FEE4E2")),
    "warn": ("CHECK", colors.HexColor("#B54708"), colors.HexColor("#FEF0C7")),
    "pass": ("PASSED", colors.HexColor("#067647"), colors.HexColor("#D1FADF")),
    "na": ("N/A", colors.HexColor("#475467"), colors.HexColor("#EAECF0")),
}
VERDICT_STYLE = {
    "review": (colors.HexColor("#B42318"), colors.HexColor("#FEE4E2")),
    "unclear": (colors.HexColor("#475467"), colors.HexColor("#EAECF0")),
    "attention": (colors.HexColor("#B54708"), colors.HexColor("#FEF0C7")),
    "clear": (colors.HexColor("#067647"), colors.HexColor("#D1FADF")),
}
EXTRA_ACTIONS = {
    "date_sequence": [
        "Confirm the invoice, due and payment dates on the original document."],
    "currency": [
        "Confirm the invoice currency before approving payment."],
}


def _money(value, currency):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{currency + ' ' if currency else ''}{float(value):,.2f}"


def _styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("t", parent=base["Title"], fontSize=22,
                                leading=26, alignment=0, textColor=NAVY,
                                spaceAfter=2)
    s["subtitle"] = ParagraphStyle("st", parent=base["BodyText"], fontSize=10,
                                   textColor=MUTED, spaceAfter=10)
    s["h2"] = ParagraphStyle("h2", parent=base["Heading2"], fontSize=13,
                             leading=17, textColor=NAVY, spaceBefore=14,
                             spaceAfter=6)
    s["h3"] = ParagraphStyle("h3", parent=base["Heading3"], fontSize=10.5,
                             leading=14, textColor=INK, spaceBefore=8,
                             spaceAfter=3)
    s["body"] = ParagraphStyle("b", parent=base["BodyText"], fontSize=9,
                               leading=13, textColor=INK)
    s["small"] = ParagraphStyle("sm", parent=base["BodyText"], fontSize=8,
                                leading=11, textColor=INK)
    s["muted"] = ParagraphStyle("mu", parent=base["BodyText"], fontSize=7.5,
                                leading=10.5, textColor=MUTED)
    s["cellb"] = ParagraphStyle("cb", parent=s["small"], fontName="Helvetica-Bold")
    s["banner"] = ParagraphStyle("bn", parent=base["BodyText"], fontSize=15,
                                 leading=19, fontName="Helvetica-Bold")
    return s


def _grid(extra=None):
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    return TableStyle(style + (extra or []))


def _header_style():
    return [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
    ]


def _p(text, style):
    return Paragraph(escape(str(text)), style)


def _page_decorations(invoice_name):
    def draw(canvas, doc):
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 8.5 * mm,
                          f"TaxPulse invoice assessment  |  {str(invoice_name)[:60]}")
        canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def create_pdf_report(analysis):
    """Build the PDF report for one analysis dict and return its bytes.

    ``analysis`` is produced by ``taxpulse_pipeline.analyze_invoice``.
    """
    record = analysis["record"]
    findings = analysis["rule_findings"]
    level, headline, detail = analysis["verdict"]
    ml_results = analysis.get("ml_results")
    feature_row = analysis.get("feature_row")
    cur = record.get("document_currency")
    name = analysis.get("file_name") or record.get("file_name") or "invoice.pdf"
    st = _styles()

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title="TaxPulse Invoice Assessment", author="TaxPulse",
    )
    width = A4[0] - 36 * mm
    story = []

    # ---- Title and verdict ---------------------------------------------
    story.append(Paragraph("TaxPulse", st["title"]))
    generated = analysis.get("generated_at") or datetime.now().strftime("%d %b %Y, %H:%M")
    story.append(Paragraph(
        escape(f"Invoice discrepancy assessment  |  {name}  |  {generated}"),
        st["subtitle"]))

    fg, bg = VERDICT_STYLE.get(level, VERDICT_STYLE["unclear"])
    banner = Table([[
        Paragraph(f'<font color="{fg.hexval().replace("0x", "#")}">{escape(headline)}</font>',
                  st["banner"]),
    ], [_p(detail, st["body"])]], colWidths=[width])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (0, 0), 10),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
        ("LINEBEFORE", (0, 0), (0, -1), 4, fg),
    ]))
    story.append(banner)
    story.append(Spacer(1, 4 * mm))

    counts = {k: sum(1 for f in findings if f["status"] == k)
              for k in ("fail", "warn", "pass", "na")}
    ml_flagged = sum(1 for r in ml_results if r["flagged"]) if ml_results else None
    tiles = [[
        Paragraph(f'<b>{counts["fail"]}</b><br/>Failed', st["body"]),
        Paragraph(f'<b>{counts["warn"]}</b><br/>To confirm', st["body"]),
        Paragraph(f'<b>{counts["pass"]}</b><br/>Passed', st["body"]),
        Paragraph(f'<b>{counts["na"]}</b><br/>Not evaluable', st["body"]),
        Paragraph(f'<b>{ml_flagged if ml_flagged is not None else "-"}</b><br/>ML signals', st["body"]),
    ]]
    tile_table = Table(tiles, colWidths=[width / 5] * 5)
    tile_table.setStyle(_grid([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                               ("TOPPADDING", (0, 0), (-1, -1), 7),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(tile_table)

    # ---- 1. Invoice details --------------------------------------------
    story.append(Paragraph("1. Invoice details", st["h2"]))
    detail_rows = [
        ("Supplier", record.get("supplier_name")),
        ("Supplier address", record.get("supplier_address")),
        ("Supplier contact", " / ".join(v for v in (
            record.get("supplier_phone"), record.get("supplier_email")) if v) or None),
        ("Registration", record.get("supplier_company_registrations")),
        ("Invoice number", record.get("invoice_number")),
        ("PO / reference", record.get("po_number") or record.get("reference_numbers")),
        ("Invoice date", record.get("document_date")),
        ("Due date", record.get("due_date")),
        ("Currency", cur),
        ("Subtotal", _money(record.get("document_total_net"), cur)
         if record.get("document_total_net") is not None else None),
        ("Tax", _money(record.get("document_total_tax"), cur)
         if record.get("document_total_tax") is not None else None),
        ("Total", _money(record.get("document_total_amount"), cur)
         if record.get("document_total_amount") is not None else None),
    ]
    rows = [[_p(k, st["cellb"]), _p(v if v else "Not found", st["small"])]
            for k, v in detail_rows]
    details_table = Table(rows, colWidths=[38 * mm, width - 38 * mm])
    details_table.setStyle(_grid([("BACKGROUND", (0, 0), (0, -1),
                                   colors.HexColor("#F2F4F7"))]))
    story.append(details_table)

    items = record.get("line_items") or []
    if items:
        story.append(Paragraph("Line items", st["h3"]))
        item_rows = [["Description", "Qty", "Unit price", "Amount", "Tax rate", "Tax"]]
        for item in items:
            item_rows.append([
                _p(item.get("description", ""), st["small"]),
                f'{item.get("quantity", 0):g}',
                _money(item.get("unit_price"), None),
                _money(item.get("amount"), None),
                item.get("tax_rate") or "-",
                _money(item.get("tax_amount"), None),
            ])
        item_table = Table(item_rows, repeatRows=1, colWidths=[
            width - 112 * mm, 14 * mm, 25 * mm, 25 * mm, 20 * mm, 28 * mm])
        item_table.setStyle(_grid(_header_style() + [
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("FONTSIZE", (0, 1), (-1, -1), 8)]))
        story.append(item_table)
        if cur:
            story.append(Paragraph(f"Amounts in {cur}.", st["muted"]))

    # ---- 2. Rule checks --------------------------------------------------
    story.append(Paragraph("2. Arithmetic and completeness checks", st["h2"]))
    story.append(Paragraph(
        "These checks recalculate the invoice figures exactly. They are the "
        "authoritative part of this report.", st["body"]))
    story.append(Spacer(1, 2 * mm))

    issues = [f for f in findings if f["status"] in ("fail", "warn")]
    if issues:
        issue_rows = [["Status", "Check", "Expected", "Found"]]
        style_cmds = []
        for index, f in enumerate(issues, start=1):
            label, fg_c, bg_c = STATUS_STYLE[f["status"]]
            issue_rows.append([
                Paragraph(f'<font color="{fg_c.hexval().replace("0x", "#")}"><b>{label}</b></font>',
                          st["small"]),
                Paragraph(f'<b>{escape(f["title"])}: {escape(f["check"])}</b><br/>'
                          f'{escape(f["explanation"])}', st["small"]),
                _p(f["expected"] or "-", st["small"]),
                _p(f["actual"] or "-", st["small"]),
            ])
            style_cmds.append(("BACKGROUND", (0, index), (0, index), bg_c))
        issue_table = Table(issue_rows, repeatRows=1,
                            colWidths=[17 * mm, width - 17 * mm - 76 * mm, 38 * mm, 38 * mm])
        issue_table.setStyle(_grid(_header_style() + style_cmds))
        story.append(issue_table)
    else:
        story.append(Paragraph("No failed checks or items to confirm.", st["body"]))

    passed = [f for f in findings if f["status"] == "pass"]
    skipped = [f for f in findings if f["status"] == "na"]
    if passed:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            "<b>Passed:</b> " + escape("; ".join(
                f'{f["title"]} ({f["check"]})' for f in passed)), st["small"]))
    if skipped:
        story.append(Spacer(1, 1 * mm))
        story.append(Paragraph(
            "<b>Could not be evaluated:</b> " + escape("; ".join(
                f'{f["title"]} ({f["check"]})' for f in skipped)), st["small"]))

    # ---- 3. Recommended actions -----------------------------------------
    actions, seen = [], set()
    for f in issues:
        steps = (CATEGORY_DETAILS.get(f["category"], {}).get("checks")
                 or EXTRA_ACTIONS.get(f["category"], []))
        for step in steps[:2]:
            if step not in seen:
                seen.add(step)
                actions.append(step)
    if analysis.get("warnings"):
        actions.append("Compare the extracted fields with the original PDF: "
                       "some fields could not be read automatically.")
    if actions:
        story.append(Paragraph("3. Recommended actions", st["h2"]))
        for step in actions:
            story.append(Paragraph("&bull; " + escape(step), st["body"]))

    # ---- 4. ML assessment -------------------------------------------------
    story.append(Paragraph("4. Pattern-based risk signals (machine learning)", st["h2"]))
    if ml_results:
        story.append(Paragraph(
            "Gaussian Naive Bayes models estimate how closely this invoice's "
            "features resemble invoices with each discrepancy type in the "
            "training data. Treat these as secondary signals: they were "
            "trained on synthetic data and have limited accuracy for several "
            "categories.", st["body"]))
        story.append(Spacer(1, 2 * mm))
        ml_rows = [["Category", "Probability", "Threshold", "Signal"]]
        ml_style = []
        for index, r in enumerate(ml_results, start=1):
            if r.get("probability") is None:
                ml_rows.append([_p(r["title"], st["small"]), "n/a",
                                f'{r["threshold"]:.1%}', "Not evaluable"])
                continue
            ml_rows.append([_p(r["title"], st["small"]),
                            f'{r["probability"]:.1%}', f'{r["threshold"]:.1%}',
                            "Flagged" if r["flagged"] else "Not flagged"])
            if r["flagged"]:
                ml_style.append(("BACKGROUND", (3, index), (3, index),
                                 STATUS_STYLE["warn"][2]))
        ml_table = Table(ml_rows, repeatRows=1,
                         colWidths=[width - 90 * mm, 30 * mm, 30 * mm, 30 * mm])
        ml_table.setStyle(_grid(_header_style() + ml_style + [
            ("FONTSIZE", (0, 1), (-1, -1), 8), ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
        story.append(ml_table)

        for r in ml_results:
            if not r["flagged"] or not r.get("evidence"):
                continue
            block = [Paragraph(f'<b>{escape(r["title"])}</b>: strongest feature evidence',
                               st["h3"])]
            ev_rows = [["Feature", "Value", "Leans toward"]]
            for item in r["evidence"]:
                value = item.get("value")
                shown = f"{float(value):.4g}" if value is not None and np.isfinite(float(value)) else "-"
                ev_rows.append([_p(item.get("label", item.get("feature", "")), st["small"]),
                                shown, item.get("direction", "-")])
            ev_table = Table(ev_rows, repeatRows=1, colWidths=[width - 80 * mm, 25 * mm, 55 * mm])
            ev_table.setStyle(_grid([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF4")),
                                     ("FONTSIZE", (0, 0), (-1, -1), 8)]))
            block.append(ev_table)
            story.append(KeepTogether(block))
    else:
        story.append(Paragraph(
            "The trained model bundle was not available for this run, so only "
            "the arithmetic and completeness checks above were performed."
            + (f" ({escape(str(analysis['ml_error']))})" if analysis.get("ml_error") else ""),
            st["body"]))

    # ---- 5. Extraction notes ---------------------------------------------
    if analysis.get("warnings"):
        story.append(Paragraph("5. Extraction notes", st["h2"]))
        for w in analysis["warnings"]:
            story.append(Paragraph("&bull; " + escape(w), st["small"]))

    # ---- Appendix --------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Appendix A. Numerical features used by the models", st["h2"]))
    if feature_row is not None and len(feature_row):
        cells = []
        for column in feature_row.columns:
            value = feature_row.iloc[0][column]
            cells.append((readable_name(column),
                          "Missing" if pd.isna(value) else f"{float(value):.6g}"))
        half = (len(cells) + 1) // 2
        left, right = cells[:half], cells[half:]
        rows = [["Feature", "Value", "Feature", "Value"]]
        for i in range(half):
            l = left[i]
            r = right[i] if i < len(right) else ("", "")
            rows.append([_p(l[0], st["muted"]), l[1], _p(r[0], st["muted"]), r[1]])
        feature_table = Table(rows, repeatRows=1, colWidths=[
            (width - 40 * mm) / 2, 20 * mm, (width - 40 * mm) / 2, 20 * mm])
        feature_table.setStyle(_grid(_header_style() + [("FONTSIZE", (0, 1), (-1, -1), 7)]))
        story.append(feature_table)

    story.append(Paragraph("Appendix B. Method and limitations", st["h2"]))
    for item in [
        "Arithmetic checks use a tolerance of 0.02 plus 0.01 per line item to allow for rounding.",
        "Values are extracted automatically from the PDF text. Extraction errors "
        "can cause false alarms or hide real ones; verify key amounts against the original.",
        "Machine-learning probabilities are model estimates, not proof that a "
        "discrepancy exists or is absent, and they do not indicate fraud.",
        "Checks that need external records (purchase orders, supplier master "
        "data, earlier invoices for duplicate detection) are not performed.",
        "Models were trained on synthetic invoice data; validate performance "
        "on real invoices before relying on them.",
        "This report supports human review. It does not authorise payment or "
        "replace accounting, tax or compliance review.",
    ]:
        story.append(Paragraph("&bull; " + escape(item), st["small"]))
        story.append(Spacer(1, 1.2 * mm))

    decorate = _page_decorations(name)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()
