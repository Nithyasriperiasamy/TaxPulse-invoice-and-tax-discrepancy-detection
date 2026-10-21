"""Deterministic invoice checks.

Most TaxPulse discrepancy categories are arithmetic identities (quantity x
price = amount, subtotal + tax = total ...).  These are checked exactly here
and are the authoritative result.  The Gaussian Naive Bayes models add a
secondary pattern-based risk signal on top.
"""

import re

RULE_CATEGORIES = {
    "line_item_arithmetic": "Line-item arithmetic",
    "invoice_total_mismatch": "Invoice total",
    "tax_amount_mismatch": "Tax amount",
    "tax_breakdown_mismatch": "Tax breakdown",
    "incomplete_supplier_details": "Supplier details",
    "document_reference_inconsistency": "Document references",
    "date_sequence": "Date sequence",
    "currency": "Currency",
}

STATUS_ORDER = {"fail": 3, "warn": 2, "pass": 1, "na": 0}
BASE_TOLERANCE = 0.02


def _fmt(value, currency):
    if value is None:
        return "Not extracted"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{value:,.2f}"


def _rate_to_float(rate):
    if rate is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(rate))
    return float(match.group(0)) / 100 if match else None


def _finding(category, check, status, severity, expected="", actual="",
             difference="", explanation=""):
    return {
        "category": category,
        "title": RULE_CATEGORIES[category],
        "check": check,
        "status": status,
        "severity": severity,
        "expected": expected,
        "actual": actual,
        "difference": difference,
        "explanation": explanation,
    }


def run_rule_checks(record):
    """Return a list of findings for one parsed invoice record."""
    cur = record.get("document_currency")
    net = record.get("document_total_net")
    tax = record.get("document_total_tax")
    total = record.get("document_total_amount")
    items = record.get("line_items") or []
    taxes = record.get("taxes") or []
    tol = BASE_TOLERANCE + 0.01 * len(items)  # allow per-line rounding

    out = []

    # ---- 1. Line-item arithmetic ---------------------------------------
    evaluable = 0
    for item in items:
        q, p, a = item.get("quantity"), item.get("unit_price"), item.get("amount")
        if None in (q, p, a):
            continue
        evaluable += 1
        expected = round(q * p, 2)
        diff = round(a - expected, 2)
        if abs(diff) > BASE_TOLERANCE:
            out.append(_finding(
                "line_item_arithmetic",
                f"{item.get('description', 'Line item')}: quantity x unit price",
                "fail", "high",
                _fmt(expected, cur), _fmt(a, cur), _fmt(abs(diff), cur),
                f"{q:g} x {_fmt(p, cur)} = {_fmt(expected, cur)}, but the "
                f"invoice prints {_fmt(a, cur)} "
                f"({'higher' if diff > 0 else 'lower'} by {_fmt(abs(diff), cur)}).",
            ))
    if evaluable == 0:
        out.append(_finding(
            "line_item_arithmetic", "Quantity x unit price", "na", "info",
            explanation="No line items with quantity, unit price and amount "
                        "could be extracted, so this check could not run."))
    elif not any(f["category"] == "line_item_arithmetic" and f["status"] == "fail"
                 for f in out):
        out.append(_finding(
            "line_item_arithmetic", "Quantity x unit price", "pass", "info",
            explanation=f"All {evaluable} line items multiply out correctly."))

    amounts = [i.get("amount") for i in items if i.get("amount") is not None]
    line_sum = round(sum(amounts), 2) if amounts else None
    if line_sum is not None and net is not None:
        diff = round(line_sum - net, 2)
        if abs(diff) > tol:
            out.append(_finding(
                "line_item_arithmetic", "Line amounts vs subtotal", "fail", "high",
                _fmt(net, cur), _fmt(line_sum, cur), _fmt(abs(diff), cur),
                "The line amounts do not add up to the printed subtotal."))
        else:
            out.append(_finding(
                "line_item_arithmetic", "Line amounts vs subtotal", "pass", "info",
                explanation="Line amounts add up to the subtotal."))

    # ---- 2. Invoice total ------------------------------------------------
    if None not in (net, tax, total):
        expected = round(net + tax, 2)
        diff = round(total - expected, 2)
        if abs(diff) > tol:
            out.append(_finding(
                "invoice_total_mismatch", "Subtotal + tax = total", "fail", "high",
                _fmt(expected, cur), _fmt(total, cur), _fmt(abs(diff), cur),
                f"Subtotal {_fmt(net, cur)} + tax {_fmt(tax, cur)} = "
                f"{_fmt(expected, cur)}, but the invoice total is "
                f"{_fmt(total, cur)}."))
        else:
            out.append(_finding(
                "invoice_total_mismatch", "Subtotal + tax = total", "pass", "info",
                explanation="Subtotal plus tax equals the printed total."))
    else:
        out.append(_finding(
            "invoice_total_mismatch", "Subtotal + tax = total", "na", "info",
            explanation="Subtotal, tax or total was not extracted."))

    # ---- 3. Tax amount ---------------------------------------------------
    tax_values = [i.get("tax_amount") for i in items if i.get("tax_amount") is not None]
    line_tax_sum = round(sum(tax_values), 2) if tax_values else None
    tax_checked = False

    if line_tax_sum is not None and tax is not None:
        tax_checked = True
        diff = round(line_tax_sum - tax, 2)
        if abs(diff) > tol:
            out.append(_finding(
                "tax_amount_mismatch", "Line-item tax vs invoice tax", "fail", "high",
                _fmt(tax, cur), _fmt(line_tax_sum, cur), _fmt(abs(diff), cur),
                "The tax on the individual lines does not add up to the "
                "invoice-level tax."))
        else:
            out.append(_finding(
                "tax_amount_mismatch", "Line-item tax vs invoice tax", "pass", "info",
                explanation="Line-item tax adds up to the invoice tax."))

    rated = [(row.get("taxable_base"), _rate_to_float(row.get("tax_rate")))
             for row in taxes]
    rated = [(b, rt) for b, rt in rated if b is not None and rt is not None]
    if rated and tax is not None and len(rated) == len(taxes):
        tax_checked = True
        expected = round(sum(b * rt for b, rt in rated), 2)
        diff = round(tax - expected, 2)
        label = " + ".join(f"{t.get('tax_rate')} of {_fmt(t.get('taxable_base'), cur)}"
                           for t in taxes)
        if abs(diff) > tol:
            out.append(_finding(
                "tax_amount_mismatch", "Tax vs taxable base x rate", "fail", "high",
                _fmt(expected, cur), _fmt(tax, cur), _fmt(abs(diff), cur),
                f"{label} should give {_fmt(expected, cur)} tax; the invoice "
                f"shows {_fmt(tax, cur)}."))
        else:
            out.append(_finding(
                "tax_amount_mismatch", "Tax vs taxable base x rate", "pass", "info",
                explanation="Invoice tax equals taxable base times the stated rate."))
    if not tax_checked:
        out.append(_finding(
            "tax_amount_mismatch", "Tax recalculation", "na", "info",
            explanation="Not enough tax data was extracted to recalculate."))

    # ---- 4. Tax breakdown ------------------------------------------------
    breakdown_checked = False
    if taxes and tax is not None:
        summary_tax = sum(r["tax_amount"] for r in taxes if r.get("tax_amount") is not None)
        breakdown_checked = True
        diff = round(summary_tax - tax, 2)
        if abs(diff) > tol:
            out.append(_finding(
                "tax_breakdown_mismatch", "Tax summary vs invoice tax", "fail", "high",
                _fmt(tax, cur), _fmt(summary_tax, cur), _fmt(abs(diff), cur),
                "The tax summary table does not match the invoice tax total."))
        else:
            out.append(_finding(
                "tax_breakdown_mismatch", "Tax summary vs invoice tax", "pass", "info",
                explanation="Tax summary matches the invoice tax total."))

    bases = [row.get("taxable_base") for row in taxes]
    reference = net if net is not None else line_sum
    if bases and all(v is not None for v in bases) and reference is not None:
        breakdown_checked = True
        base_sum = round(sum(bases), 2)
        diff = round(base_sum - reference, 2)
        if abs(diff) > tol:
            out.append(_finding(
                "tax_breakdown_mismatch", "Taxable bases vs subtotal", "warn", "medium",
                _fmt(reference, cur), _fmt(base_sum, cur), _fmt(abs(diff), cur),
                "The taxable bases in the tax summary do not add up to the "
                "subtotal. This can be legitimate (exempt items, discounts) "
                "but should be confirmed."))
        else:
            out.append(_finding(
                "tax_breakdown_mismatch", "Taxable bases vs subtotal", "pass", "info",
                explanation="Taxable bases add up to the subtotal."))
    if not breakdown_checked:
        out.append(_finding(
            "tax_breakdown_mismatch", "Tax breakdown", "na", "info",
            explanation="No tax summary table was extracted."))

    # ---- 5. Supplier details --------------------------------------------
    required = {
        "name": "supplier_name", "address": "supplier_address",
        "phone": "supplier_phone", "email": "supplier_email",
        "tax/company registration": "supplier_company_registrations",
    }
    missing = [label for label, key in required.items() if not record.get(key)]
    if not record.get("supplier_name"):
        out.append(_finding(
            "incomplete_supplier_details", "Supplier identity", "fail", "high",
            "Supplier name present", "Missing", "",
            "The supplier name could not be found."))
    elif missing:
        severity = "medium" if (
            not record.get("supplier_address")
            and not record.get("supplier_company_registrations")) else "low"
        out.append(_finding(
            "incomplete_supplier_details", "Supplier completeness", "warn", severity,
            "Name, address, contact and registration", "Missing: " + ", ".join(missing), "",
            "Some supplier details were not found. Confirm whether the "
            "original invoice contains them."))
    else:
        out.append(_finding(
            "incomplete_supplier_details", "Supplier completeness", "pass", "info",
            explanation="All supplier fields were found."))

    # ---- 6. Document references -----------------------------------------
    if not record.get("invoice_number") and not record.get("document_number"):
        out.append(_finding(
            "document_reference_inconsistency", "Invoice number", "fail", "medium",
            "An invoice number", "Missing", "",
            "No invoice number was found. Without it the invoice cannot be "
            "matched or checked for duplicates."))
    else:
        out.append(_finding(
            "document_reference_inconsistency", "Invoice number", "pass", "info",
            explanation=f"Invoice number {record.get('invoice_number')} found."))
    if not record.get("po_number") and not record.get("reference_numbers"):
        out.append(_finding(
            "document_reference_inconsistency", "PO / customer reference", "warn", "low",
            "A PO or customer reference", "Missing", "",
            "No purchase-order or customer reference found. Matching against "
            "your PO records needs an external source."))

    # ---- 7. Dates ---------------------------------------------------------
    doc, due, paid = (record.get("document_date"), record.get("due_date"),
                      record.get("payment_date"))
    date_issues, date_checked = [], False
    if doc and due:
        date_checked = True
        if due < doc:
            date_issues.append(f"Due date {due} is before invoice date {doc}.")
    if doc and paid:
        date_checked = True
        if paid < doc:
            date_issues.append(f"Payment date {paid} is before invoice date {doc}.")
    if date_issues:
        out.append(_finding("date_sequence", "Date order", "fail", "medium",
                            "Invoice date <= due / payment date", "; ".join(date_issues),
                            "", " ".join(date_issues)))
    elif date_checked:
        out.append(_finding("date_sequence", "Date order", "pass", "info",
                            explanation="Dates are in a logical order."))
    else:
        out.append(_finding("date_sequence", "Date order", "na", "info",
                            explanation="Not enough dates were extracted."))

    # ---- 8. Currency ------------------------------------------------------
    if not cur:
        out.append(_finding("currency", "Currency", "warn", "low",
                            "A stated currency", "Missing", "",
                            "No currency was found on the invoice."))
    else:
        out.append(_finding("currency", "Currency", "pass", "info",
                            explanation=f"Currency {cur} found."))

    return out


def category_status(findings):
    """Collapse findings to the worst status for each category."""
    result = {}
    for f in findings:
        current = result.get(f["category"], "na")
        if STATUS_ORDER[f["status"]] > STATUS_ORDER[current]:
            current = f["status"]
        result[f["category"]] = current
    return result


def overall_verdict(findings, extraction_warnings=None):
    """Return (level, headline, detail) summarising the rule results."""
    fails = [f for f in findings if f["status"] == "fail"]
    warns = [f for f in findings if f["status"] == "warn"]
    na = [f for f in findings if f["status"] == "na"]
    core_na = {"line_item_arithmetic", "invoice_total_mismatch"} <= {
        f["category"] for f in na}

    if fails:
        cats = sorted({RULE_CATEGORIES[f["category"]] for f in fails})
        return ("review", "Review required",
                f"{len(fails)} failed check(s) in: {', '.join(cats)}.")
    if core_na:
        return ("unclear", "Could not verify",
                "The key amounts could not be extracted, so the arithmetic "
                "checks did not run. Review the extracted data.")
    if warns:
        return ("attention", "Minor items to confirm",
                f"No arithmetic errors, but {len(warns)} item(s) need confirmation.")
    return ("clear", "No discrepancies found",
            "All evaluable checks passed.")
