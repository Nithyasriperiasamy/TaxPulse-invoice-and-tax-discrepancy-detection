"""PDF invoice parser for TaxPulse.

Turns a text-based invoice PDF into the record dictionary used by the
feature extractor and rules engine.  Every value that cannot be found is
left as ``None`` and reported in ``warnings`` instead of being guessed.
"""

import re
from datetime import datetime
from io import BytesIO

import pdfplumber

CURRENCY_SYMBOLS = r"[$€£₹]"
MONEY_PATTERN = r"(?:" + CURRENCY_SYMBOLS + r")?\s*-?\d[\d,]*\.\d{2}"
SYMBOL_TO_CODE = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}

DATE_FORMATS = [
    "%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
    "%B %d, %Y", "%b %d, %Y", "%d-%m-%Y", "%d.%m.%Y",
]


# ---------------------------------------------------------------- helpers
def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_number(value):
    if value is None:
        return None
    value = str(value).strip()
    negative = value.startswith("(") and value.endswith(")")
    value = value.strip("()")
    value = re.sub(r"[$€£₹,\s]", "", value)
    try:
        number = float(value)
        return -number if negative else number
    except (ValueError, TypeError):
        return None


def parse_date(value):
    if not value:
        return None
    value = clean_text(value)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def extract_labeled_value(text, labels):
    """Find 'Label: value' (colon optional) at the start of a line."""
    for label in labels:
        pattern = (
            r"(?im)^\s*" + re.escape(label) + r"\b\s*[:#\-]?\s*(.+?)\s*$"
        )
        match = re.search(pattern, text)
        if match:
            value = clean_text(match.group(1))
            if value:
                return value
    return None


def extract_section(text, start_label, end_label):
    start = re.search(re.escape(start_label), text, flags=re.IGNORECASE)
    if not start:
        return ""
    remaining = text[start.end():]
    end = re.search(re.escape(end_label), remaining, flags=re.IGNORECASE)
    return remaining[:end.start()] if end else remaining


def _lines(text):
    return [clean_text(l) for l in text.splitlines() if clean_text(l)]


# --------------------------------------------------------------- supplier
def extract_supplier_name(text):
    skip = re.compile(
        r"^(?:tax\s+)?invoice$|^invoice\b|^bill\b|^page\b", re.IGNORECASE
    )
    for line in _lines(text)[:8]:
        if not skip.search(line) and re.search(r"[A-Za-z]", line):
            return line
    return None


def extract_supplier_address(text):
    lines = _lines(text)
    street = (
        r"\b(?:avenue|ave|street|st|road|rd|boulevard|blvd|drive|dr|lane|"
        r"ln|way|suite|unit|nagar|colony|marg)\b"
    )
    start = None
    for index, line in enumerate(lines):
        if re.match(r"^\d+[\w/-]*[,\s]+\w", line) and re.search(
            street, line, re.IGNORECASE
        ):
            start = index
            break
    if start is None:
        return None

    parts = [lines[start]]
    for line in lines[start + 1:]:
        if re.search(
            r"\b(?:phone|tel|email|www\.|https?://|invoice)\b",
            line, re.IGNORECASE,
        ):
            break
        if re.match(r"^(?:due date|currency|bill to|customer)\b", line,
                    re.IGNORECASE):
            break
        parts.append(line)
        if re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", line) or re.search(
            r"\b\d{6}\b", line
        ):
            break
        if len(parts) >= 3:
            break
    return clean_text(", ".join(parts)) or None


def extract_registration(text):
    patterns = [
        r"(?im)^\s*(?:GSTIN|GST\s*(?:No|Number)?|VAT\s*(?:No|ID|Number)?|"
        r"Tax\s*ID|EIN|TIN|PAN|CIN|Company\s*(?:Reg(?:istration)?|No))"
        r"\.?\s*[:#]?\s*([A-Z0-9][A-Z0-9\-/ ]{4,})\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return clean_text(match.group(1))
    return None


def extract_website(text):
    match = re.search(
        r"(?i)(?<![@\w.])((?:https?://|www\.)[A-Z0-9.-]+\.[A-Z]{2,}[^\s]*)",
        text,
    )
    return match.group(1) if match else None


# ------------------------------------------------------------- line items
def extract_line_items(text):
    section = extract_section(text, "DESCRIPTION OF SERVICES", "TAX SUMMARY")
    if not section:
        # Fallback: look between any items-style header and the totals.
        header = re.search(
            r"(?im)^\s*(?:item|description)\b.*\b(?:qty|quantity)\b.*$", text
        )
        if header:
            section = text[header.end():]
            end = re.search(
                r"(?im)^\s*(?:tax summary|sub\s*total|invoice totals)\b",
                section,
            )
            if end:
                section = section[:end.start()]
    if not section:
        return []

    # Drop header lines only (a header line names several column titles),
    # so words such as "Amount" inside a real description are kept.
    header_words = re.compile(
        r"\b(?:item|service|description|qty|quantity|unit\s*price|price|"
        r"line\s*amount|amount|tax\s*rate|rate|tax\s*amount|tax|total)\b",
        re.IGNORECASE,
    )
    kept = []
    for raw in section.splitlines():
        if len(header_words.findall(raw)) >= 3 and not re.search(r"\d+\.\d{2}", raw):
            continue
        kept.append(raw)
    section = re.sub(r"\s+", " ", " ".join(kept)).strip()

    money = CURRENCY_SYMBOLS + r"?\s*\d[\d,]*\.\d{2}"
    qty = r"\d+(?:\.\d+)?"
    rate = r"\d+(?:\.\d+)?\s*%"
    desc = r"(?P<description>[A-Za-z][A-Za-z0-9 /&().,'-]*?)\s+"

    full = re.compile(
        desc + rf"(?P<quantity>{qty})\s+(?P<unit_price>{money})\s+"
        rf"(?P<amount>{money})\s+(?P<tax_rate>{rate})\s+"
        rf"(?P<tax_amount>{money})"
    )
    no_tax = re.compile(
        desc + rf"(?P<quantity>{qty})\s+(?P<unit_price>{money})\s+"
        rf"(?P<amount>{money})(?!\s*{rate})"
    )

    items = []
    for match in full.finditer(section):
        items.append(_item_from_match(match))
    if not items:
        for match in no_tax.finditer(section):
            items.append(_item_from_match(match))
    return [item for item in items if item]


def _item_from_match(match):
    groups = match.groupdict()
    description = clean_text(groups["description"])
    quantity = parse_number(groups["quantity"])
    unit_price = parse_number(groups["unit_price"])
    amount = parse_number(groups["amount"])
    if not description or None in (quantity, unit_price, amount):
        return None
    return {
        "description": description,
        "quantity": quantity,
        "unit_price": unit_price,
        "amount": amount,
        "tax_rate": clean_text(groups.get("tax_rate")) or None,
        "tax_amount": parse_number(groups.get("tax_amount")),
    }


# ------------------------------------------------------- tax and totals
def extract_tax_summary(text):
    """Return ``(rows, total_tax)``; one row per tax rate in the summary."""
    section = extract_section(text, "TAX SUMMARY", "INVOICE TOTALS")
    if not section:
        return [], None

    rows = []
    for raw in section.splitlines():
        amounts = re.findall(MONEY_PATTERN, raw)
        rates = re.findall(r"\d+(?:\.\d+)?\s*%", raw)
        if len(amounts) >= 2 and len(rates) == 1:
            rows.append({
                "taxable_base": parse_number(amounts[0]),
                "tax_rate": clean_text(rates[0]),
                "tax_amount": parse_number(amounts[-1]),
            })

    if not rows:  # unusual layout: fall back to the whole section
        amounts = re.findall(MONEY_PATTERN, section)
        rates = re.findall(r"\d+(?:\.\d+)?\s*%", section)
        if len(amounts) >= 2:
            rows.append({
                "taxable_base": parse_number(amounts[0]),
                "tax_rate": clean_text(rates[0]) if rates else None,
                "tax_amount": parse_number(amounts[-1]),
            })

    total = round(sum(r["tax_amount"] for r in rows), 2) if rows else None
    return rows, total


def extract_invoice_totals(text):
    section = extract_section(text, "INVOICE TOTALS", "NOTES AND TERMS")
    if not section:
        section = text  # fall back to the whole document

    money = MONEY_PATTERN

    def find(pattern):
        match = re.search(pattern, section)
        return parse_number(match.group(1)) if match else None

    subtotal = find(
        rf"(?im)^\s*(?:Sub[\s-]?total|Net\s+(?:Amount|Total))\s*:?\s*({money})\s*$"
    )
    tax = find(
        rf"(?im)^\s*(?:Total\s+)?(?:Sales\s+)?(?:Tax|VAT|GST)"
        rf"(?:\s*\([^)]*\))?\s*:?\s*({money})\s*$"
    )
    total = find(
        rf"(?is)(?:TOTAL\s+AMOUNT\s+DUE|GRAND\s+TOTAL|AMOUNT\s+DUE|"
        rf"^\s*TOTAL)\s*:?\s*({money})"
    )
    if total is None:
        total = find(rf"(?im)^\s*Total\s*:?\s*({money})\s*$")

    # Positional fallback only inside a real totals block.
    if section is not text:
        amounts = [parse_number(v) for v in re.findall(money, section)]
        amounts = [v for v in amounts if v is not None]
        if subtotal is None and amounts:
            subtotal = amounts[0]
        if tax is None and len(amounts) >= 2:
            tax = amounts[1]
        if total is None and amounts:
            total = amounts[-1]

    return subtotal, tax, total


def detect_currency(text):
    match = re.search(r"\b(USD|EUR|GBP|INR|CAD|AUD|SGD|JPY)\b", text,
                      re.IGNORECASE)
    if match:
        return match.group(1).upper(), False
    symbol = re.search(CURRENCY_SYMBOLS, text)
    if symbol:
        return SYMBOL_TO_CODE[symbol.group(0)], True
    return None, False


# ----------------------------------------------------------------- main
def parse_invoice_pdf(source, file_name=None):
    """Parse a text-based invoice PDF.

    ``source`` may be a path, a file-like object (Streamlit upload) or raw
    bytes.  Returns ``(record, extracted_text, warnings)``.
    """
    if isinstance(source, (bytes, bytearray)):
        stream = BytesIO(bytes(source))
    else:
        stream = source
        if hasattr(stream, "seek"):
            stream.seek(0)
    file_name = file_name or getattr(source, "name", None)

    try:
        with pdfplumber.open(stream) as pdf:
            page_texts = [page.extract_text() or "" for page in pdf.pages]
    except Exception as error:
        raise ValueError(
            "The file could not be read as a PDF. It may be corrupted or "
            f"password protected. ({error})"
        ) from error

    text = "\n".join(page_texts).strip()
    if not text:
        raise ValueError(
            "No selectable text was found. This PDF looks scanned and "
            "needs OCR before it can be analysed."
        )

    warnings = []

    supplier_name = extract_supplier_name(text)
    supplier_address = extract_supplier_address(text)
    supplier_phone = extract_labeled_value(text, ["Phone", "Telephone", "Tel"])
    email_match = re.search(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.IGNORECASE
    )

    invoice_number = extract_labeled_value(
        text, ["Invoice Number", "Invoice No", "Invoice #", "Document Number"]
    )
    po_number = extract_labeled_value(
        text, ["Purchase Order Number", "Purchase Order", "PO Number", "PO #"]
    )
    reference = extract_labeled_value(
        text, ["Customer Reference", "Reference Number", "Ref"]
    )

    document_date = parse_date(extract_labeled_value(
        text, ["Invoice Date", "Document Date", "Date of Issue"]))
    due_date = parse_date(extract_labeled_value(
        text, ["Due Date", "Payment Due Date"]))
    payment_date = parse_date(extract_labeled_value(
        text, ["Payment Date", "Date Paid"]))

    # dd/mm vs mm/dd cannot be told apart when both parts are 12 or less.
    for label in ("Invoice Date", "Document Date", "Due Date", "Payment Date"):
        raw = extract_labeled_value(text, [label])
        m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/\d{4}", raw or "")
        if m and int(m.group(1)) <= 12 and int(m.group(2)) <= 12 \
                and m.group(1) != m.group(2):
            warnings.append(
                f"{label} '{raw}' is ambiguous (day/month or month/day). "
                "It was read as month/day; check it on the original.")

    currency, inferred = detect_currency(text)
    line_items = extract_line_items(text)
    taxes, summary_tax = extract_tax_summary(text)
    subtotal, totals_tax, total_amount = extract_invoice_totals(text)
    total_tax = totals_tax if totals_tax is not None else summary_tax

    checks = [
        (supplier_address is None, "Supplier address was not identified."),
        (supplier_phone is None, "Supplier phone number was not identified."),
        (email_match is None, "Supplier email was not identified."),
        (invoice_number is None, "Invoice number was not identified."),
        (not line_items, "No line-item table was identified."),
        (subtotal is None, "Invoice subtotal was not identified."),
        (total_tax is None, "Tax amount was not identified."),
        (total_amount is None, "Invoice total was not identified."),
        (currency is None, "Currency was not identified."),
        (document_date is None, "Invoice date was not identified."),
    ]
    warnings.extend(message for missing, message in checks if missing)
    if inferred:
        warnings.append(
            f"Currency was inferred from the symbol as {currency}; "
            "no currency code was printed."
        )

    record = {
        "file_name": file_name,
        "supplier_name": supplier_name,
        "supplier_address": supplier_address,
        "supplier_phone": supplier_phone,
        "supplier_email": email_match.group(0) if email_match else None,
        "supplier_website": extract_website(text),
        "supplier_company_registrations": extract_registration(text),
        "document_number": invoice_number,
        "invoice_number": invoice_number,
        "receipt_number": None,
        "po_number": po_number,
        "reference_numbers": reference,
        "document_date": document_date,
        "due_date": due_date,
        "payment_date": payment_date,
        "document_currency": currency,
        "document_total_net": subtotal,
        "document_total_tax": total_tax,
        "document_total_amount": total_amount,
        "line_items": line_items,
        "taxes": taxes,
        "document_type_extended": "invoice",
        "document_category": None,
        "subcategory": None,
        "document_time": None,
        "customer_name": None,
        "customer_id": None,
        "customer_address": None,
        "billing_address": None,
        "shipping_address": None,
        "customer_company_registrations": None,
        "locale": None,
        "notes": None,
        "mime_type": "application/pdf",
    }
    return record, text, warnings
