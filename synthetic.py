"""Synthetic invoices with known, isolated faults.

One generator feeds three things so they stay consistent:
  * train_models.py  - training data for the ML models
  * evaluate.py      - end-to-end accuracy on real PDFs (parse -> rules -> ML)
  * make_sample_invoices.py - demo PDFs

Each fault changes only the fields that define its category, so the
ground-truth label is unambiguous:

  line_item_arithmetic            one printed line amount is wrong
  tax_amount_mismatch             one line is taxed wrongly; totals follow it,
                                  so tax differs from taxable base x rate
  invoice_total_mismatch          printed total differs from subtotal + tax
  tax_breakdown_mismatch          tax summary table differs from invoice tax
  incomplete_supplier_details     1-3 of address/phone/email/registration absent
  document_reference_inconsistency  invoice number absent
"""

import copy
from io import BytesIO

import numpy as np

CATEGORIES = [
    "tax_amount_mismatch",
    "invoice_total_mismatch",
    "incomplete_supplier_details",
    "line_item_arithmetic",
    "tax_breakdown_mismatch",
    "document_reference_inconsistency",
]

SUPPLIERS = [
    "NORTHSTAR EVENT SERVICES LLC", "Harbor Light Productions Ltd",
    "Summit Stagecraft Inc.", "BlueRiver Audio Visual Co.",
    "Pinnacle Events Group", "Redwood Hospitality Services",
]
STREETS = ["Harbor Avenue", "Maple Street", "Station Road", "Oak Boulevard",
           "Mill Lane", "Cedar Drive"]
CITIES = [("Austin", "TX", "78701"), ("Denver", "CO", "80202"),
          ("Portland", "OR", "97205"), ("Boston", "MA", "02110")]
DESCRIPTIONS = [
    "Stage setup and lighting", "Sound engineer", "LED video wall rental",
    "Catering (per head)", "Security staff & marshals", "Event signage",
    "Equipment transport", "Generator hire", "Photography - full day",
    "Floor plan design", "Wireless microphone kit", "Stage crew",
]
REG_LABELS = ["GSTIN", "VAT No", "EIN", "Company Reg"]
CURRENCIES = {"USD": "$", "EUR": "€", "GBP": "£"}
SUPPLIER_FIELDS = ["supplier_address", "supplier_phone", "supplier_email",
                   "supplier_company_registrations"]


def _sign(rng):
    return 1 if rng.random() < 0.5 else -1


# --------------------------------------------------------------- generation
def clean_record(rng):
    rates = rng.choice([5, 8, 10, 12, 18], size=int(rng.integers(1, 3)),
                       replace=False)
    items = []
    for _ in range(int(rng.integers(1, 9))):
        qty = float(rng.integers(1, 21)) if rng.random() < 0.85 else \
            float(rng.integers(1, 20)) + 0.5
        price = round(float(rng.uniform(8, 900)), 2)
        rate = float(rng.choice(rates))
        amount = round(qty * price, 2)
        items.append({
            "description": str(rng.choice(DESCRIPTIONS)),
            "quantity": qty, "unit_price": price, "amount": amount,
            "tax_rate": f"{rate:g}%",
            "tax_amount": round(amount * rate / 100, 2),
        })

    taxes = []
    for rate in sorted({i["tax_rate"] for i in items}):
        group = [i for i in items if i["tax_rate"] == rate]
        taxes.append({
            "taxable_base": round(sum(i["amount"] for i in group), 2),
            "tax_rate": rate,
            "tax_amount": round(sum(i["tax_amount"] for i in group), 2),
        })

    net = round(sum(i["amount"] for i in items), 2)
    tax = round(sum(i["tax_amount"] for i in items), 2)
    city, state, zip_code = CITIES[int(rng.integers(len(CITIES)))]
    code = str(rng.choice(list(CURRENCIES)))
    month, day = int(rng.integers(1, 12)), int(rng.integers(1, 28))

    return {
        "supplier_name": str(rng.choice(SUPPLIERS)),
        "supplier_address": (f"{int(rng.integers(10, 999))} "
                             f"{rng.choice(STREETS)}, {city}, {state} {zip_code}"),
        "supplier_phone": f"({int(rng.integers(200, 999))}) 555-{int(rng.integers(1000, 9999))}",
        "supplier_email": f"billing{int(rng.integers(1, 99))}@example-events.com",
        "supplier_company_registrations": (
            f"{rng.choice(REG_LABELS)}: {int(rng.integers(10**7, 10**8))}AB1Z5"),
        "invoice_number": f"INV-2024-{int(rng.integers(1000, 9999))}",
        "po_number": f"PO-{int(rng.integers(10000, 99999))}" if rng.random() < .7 else None,
        "reference_numbers": f"CR-{int(rng.integers(1000, 9999))}" if rng.random() < .5 else None,
        "document_date": f"2024-{month:02d}-{day:02d}",
        "due_date": f"2024-{min(month + 1, 12):02d}-{day:02d}",
        "document_currency": code,
        "document_total_net": net, "document_total_tax": tax,
        "document_total_amount": round(net + tax, 2),
        "line_items": items, "taxes": taxes,
    }


def inject_faults(record, rng, p=0.25):
    """Return ``(record, faults)`` where faults maps category -> 0/1."""
    rec = copy.deepcopy(record)
    faults = dict.fromkeys(CATEGORIES, 0)
    items, taxes = rec["line_items"], rec["taxes"]

    if rng.random() < p:  # wrong printed line amount (subtotal stays as is)
        item = items[int(rng.integers(len(items)))]
        delta = max(0.5, rng.uniform(0.02, 0.15) * item["amount"])
        item["amount"] = round(item["amount"] + _sign(rng) * delta, 2)
        faults["line_item_arithmetic"] = 1

    if rng.random() < p:  # one line taxed wrongly; every total follows it
        item = items[int(rng.integers(len(items)))]
        delta = max(0.5, rng.uniform(0.1, 0.5) * max(item["tax_amount"], 1))
        item["tax_amount"] = round(max(0.0, item["tax_amount"] + _sign(rng) * delta), 2)
        for row in taxes:
            if row["tax_rate"] == item["tax_rate"]:
                row["tax_amount"] = round(sum(
                    i["tax_amount"] for i in items if i["tax_rate"] == row["tax_rate"]), 2)
        rec["document_total_tax"] = round(sum(i["tax_amount"] for i in items), 2)
        rec["document_total_amount"] = round(
            rec["document_total_net"] + rec["document_total_tax"], 2)
        faults["tax_amount_mismatch"] = 1

    if rng.random() < p:  # printed total is off
        rec["document_total_amount"] = round(
            rec["document_total_amount"] + _sign(rng) * rng.uniform(1, 200), 2)
        faults["invoice_total_mismatch"] = 1

    if rng.random() < p:  # tax summary table disagrees with invoice tax
        row = taxes[int(rng.integers(len(taxes)))]
        row["tax_amount"] = round(
            max(0.0, row["tax_amount"] + _sign(rng) * rng.uniform(0.5, 50)), 2)
        faults["tax_breakdown_mismatch"] = 1

    if rng.random() < p:  # supplier details missing
        for key in rng.choice(SUPPLIER_FIELDS, size=int(rng.integers(1, 4)),
                              replace=False):
            rec[str(key)] = None
        faults["incomplete_supplier_details"] = 1

    if rng.random() < p:  # invoice number missing
        rec["invoice_number"] = None
        faults["document_reference_inconsistency"] = 1

    # Harmless rounding noise: must NOT be treated as a fault.
    if rng.random() < 0.3 and not faults["tax_amount_mismatch"]:
        item = items[int(rng.integers(len(items)))]
        item["tax_amount"] = round(item["tax_amount"] + _sign(rng) * 0.01, 2)

    # The parsed record carries the invoice number under both keys.
    rec["document_number"] = rec["invoice_number"]
    return rec, faults


def make_dataset(n, seed, p=0.25):
    rng = np.random.default_rng(seed)
    return [inject_faults(clean_record(rng), rng, p) for _ in range(n)]


# ---------------------------------------------------------------- PDF writer
def write_invoice_pdf(record, target):
    """Render a record in the layout the parser understands.

    ``target`` is a path or a binary file-like object.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    sym = CURRENCIES.get(record.get("document_currency"), "$")
    money = lambda v: f"{sym}{v:,.2f}"
    c = canvas.Canvas(target, pagesize=A4)
    y = A4[1] - 60

    def line(text, x=50, size=10, bold=False, step=15):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, text)
        y -= step

    line(record["supplier_name"], size=16, bold=True, step=22)
    if record.get("supplier_address"):
        street, *rest = record["supplier_address"].split(", ", 1)
        line(street)
        if rest:
            line(rest[0])
    if record.get("supplier_phone"):
        line(f"Phone: {record['supplier_phone']}")
    if record.get("supplier_email"):
        line(f"Email: {record['supplier_email']}")
    if record.get("supplier_company_registrations"):
        line(record["supplier_company_registrations"])
    y -= 8
    line("INVOICE", size=13, bold=True, step=20)
    if record.get("invoice_number"):
        line(f"Invoice Number: {record['invoice_number']}")
    line(f"Invoice Date: {record['document_date']}")
    line(f"Due Date: {record['due_date']}")
    line(f"Currency: {record['document_currency']}")
    if record.get("po_number"):
        line(f"Purchase Order: {record['po_number']}")
    if record.get("reference_numbers"):
        line(f"Customer Reference: {record['reference_numbers']}")
    y -= 10

    line("DESCRIPTION OF SERVICES", bold=True, step=18)
    cols = [50, 235, 275, 345, 425, 480]
    for x, t in zip(cols, ["ITEM / SERVICE", "QTY", "UNIT PRICE",
                           "LINE AMOUNT", "TAX RATE", "TAX AMOUNT"]):
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x, y, t)
    y -= 15
    for it in record["line_items"]:
        values = [it["description"], f"{it['quantity']:g}", money(it["unit_price"]),
                  money(it["amount"]), it["tax_rate"], money(it["tax_amount"])]
        for x, v in zip(cols, values):
            c.setFont("Helvetica", 9)
            c.drawString(x, y, v)
        y -= 14
    y -= 10

    line("TAX SUMMARY", bold=True, step=18)
    for row in record["taxes"]:
        line(f"Taxable amount {money(row['taxable_base'])}    "
             f"Rate {row['tax_rate']}    Tax {money(row['tax_amount'])}")
    y -= 10
    line("INVOICE TOTALS", bold=True, step=18)
    line(f"Subtotal: {money(record['document_total_net'])}")
    line(f"Tax: {money(record['document_total_tax'])}")
    line(f"TOTAL AMOUNT DUE: {money(record['document_total_amount'])}", bold=True)
    y -= 10
    line("NOTES AND TERMS", bold=True, step=18)
    line("Payment due within 30 days. Thank you for your business.")
    c.save()


def invoice_pdf_bytes(record):
    buffer = BytesIO()
    write_invoice_pdf(record, buffer)
    return buffer.getvalue()
