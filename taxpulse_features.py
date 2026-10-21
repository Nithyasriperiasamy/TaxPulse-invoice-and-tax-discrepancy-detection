
import numpy as np
import pandas as pd


# Small tolerance for floating-point arithmetic.
# This is not a substitute for jurisdiction-specific tax rounding rules.
AMOUNT_TOLERANCE = 0.02


def extract_invoice_features(record):
    """
    Convert one extracted invoice record into numerical model features.

    Expected input:
        A dictionary returned by taxpulse_parser.py.

    Returns:
        A dictionary containing numerical features for the six
        discrepancy-detection models.

    Important:
        Missing financial values remain NaN where a reliable calculation
        is not possible. They are not automatically treated as zero.
    """

    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary.")

    line_items = record.get("line_items")
    taxes = record.get("taxes")

    if not isinstance(line_items, list):
        line_items = []

    if not isinstance(taxes, list):
        taxes = []

    def to_number(value):
        """Convert a value to a finite float or NaN."""
        if value is None:
            return np.nan

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return np.nan

            # Handle common currency formatting such as "$1,250.50".
            value = value.replace(",", "")
            value = value.replace("$", "")
            value = value.replace("€", "")
            value = value.replace("£", "")

        try:
            number = float(value)

            if not np.isfinite(number):
                return np.nan

            return number

        except (TypeError, ValueError):
            return np.nan

    def present(value):
        """Return 1 if a field contains a usable non-empty value."""
        if value is None:
            return 0

        if isinstance(value, str):
            return int(bool(value.strip()))

        if isinstance(value, (list, tuple, dict, set)):
            return int(len(value) > 0)

        try:
            return int(not bool(pd.isna(value)))
        except (TypeError, ValueError):
            return 1

    def get_date(value):
        """Convert a date to pandas Timestamp or NaT."""
        if not present(value):
            return pd.NaT

        return pd.to_datetime(value, errors="coerce")

    def absolute_difference(first, second):
        """Calculate absolute difference only when both values exist."""
        if pd.notna(first) and pd.notna(second):
            return abs(first - second)

        return np.nan

    def count_missing(values):
        return sum(pd.isna(value) for value in values)

    def safe_sum(values, require_all=True):
        """
        Sum values without silently accepting an incomplete list.

        If require_all is True, any missing value makes the result NaN.
        """
        if not values:
            return np.nan

        if require_all and any(pd.isna(value) for value in values):
            return np.nan

        valid_values = [
            value for value in values
            if pd.notna(value)
        ]

        if not valid_values:
            return np.nan

        return float(sum(valid_values))

    # ---------------------------------------------------------
    # 1. Invoice-level financial values
    # ---------------------------------------------------------

    net_amount = to_number(record.get("document_total_net"))
    tax_amount = to_number(record.get("document_total_tax"))
    total_amount = to_number(record.get("document_total_amount"))

    calculated_total = (
        net_amount + tax_amount
        if pd.notna(net_amount) and pd.notna(tax_amount)
        else np.nan
    )

    total_difference = (
        total_amount - calculated_total
        if pd.notna(total_amount) and pd.notna(calculated_total)
        else np.nan
    )

    absolute_total_difference = (
        abs(total_difference)
        if pd.notna(total_difference)
        else np.nan
    )

    # ---------------------------------------------------------
    # 2. Line-item values
    # ---------------------------------------------------------

    line_amounts = [
        to_number(item.get("amount"))
        for item in line_items
    ]

    line_taxes = [
        to_number(item.get("tax_amount"))
        for item in line_items
    ]

    quantities = [
        to_number(item.get("quantity"))
        for item in line_items
    ]

    unit_prices = [
        to_number(item.get("unit_price"))
        for item in line_items
    ]

    line_amount_sum = safe_sum(line_amounts, require_all=True)
    line_tax_sum = safe_sum(line_taxes, require_all=True)

    line_net_difference = (
        line_amount_sum - net_amount
        if pd.notna(line_amount_sum) and pd.notna(net_amount)
        else np.nan
    )

    line_tax_difference = (
        line_tax_sum - tax_amount
        if pd.notna(line_tax_sum) and pd.notna(tax_amount)
        else np.nan
    )

    absolute_line_net_difference = (
        abs(line_net_difference)
        if pd.notna(line_net_difference)
        else np.nan
    )

    absolute_line_tax_difference = (
        abs(line_tax_difference)
        if pd.notna(line_tax_difference)
        else np.nan
    )

    # ---------------------------------------------------------
    # 3. Quantity × unit price checks
    # ---------------------------------------------------------

    amount_mismatch_count = 0
    amount_absolute_error_sum = 0.0
    missing_amount_comparison_count = 0
    line_amount_checks_performed = 0

    for item in line_items:
        quantity = to_number(item.get("quantity"))
        unit_price = to_number(item.get("unit_price"))
        amount = to_number(item.get("amount"))

        if (
            pd.isna(quantity)
            or pd.isna(unit_price)
            or pd.isna(amount)
        ):
            missing_amount_comparison_count += 1
            continue

        calculated_line_amount = quantity * unit_price
        difference = amount - calculated_line_amount

        line_amount_checks_performed += 1

        if abs(difference) > AMOUNT_TOLERANCE:
            amount_mismatch_count += 1

        # Accumulate only the amount beyond the accepted tolerance.
        amount_absolute_error_sum += max(
            0.0,
            abs(difference) - AMOUNT_TOLERANCE,
        )

    # If no comparisons were possible, error sum is unavailable.
    if line_amount_checks_performed == 0:
        amount_absolute_error_sum = np.nan

    # ---------------------------------------------------------
    # 4. Missing line-item information
    # ---------------------------------------------------------

    missing_description_count = sum(
        not present(item.get("description"))
        for item in line_items
    )

    missing_line_tax_rate_count = sum(
        not present(item.get("tax_rate"))
        for item in line_items
    )

    # ---------------------------------------------------------
    # 5. Supplier information
    # ---------------------------------------------------------

    supplier_fields = [
        record.get("supplier_name"),
        record.get("supplier_address"),
        record.get("supplier_company_registrations"),
    ]

    supplier_missing_count = sum(
        not present(value)
        for value in supplier_fields
    )

    supplier_name_missing = int(
        not present(record.get("supplier_name"))
    )

    supplier_address_missing = int(
        not present(record.get("supplier_address"))
    )

    supplier_registration_missing = int(
        not present(record.get("supplier_company_registrations"))
    )

    supplier_phone_missing = int(
        not present(record.get("supplier_phone"))
    )

    supplier_email_missing = int(
        not present(record.get("supplier_email"))
    )

    supplier_website_missing = int(
        not present(record.get("supplier_website"))
    )

    supplier_contact_missing_count = sum([
        supplier_phone_missing,
        supplier_email_missing,
        supplier_website_missing,
    ])

    # ---------------------------------------------------------
    # 6. Document references
    # ---------------------------------------------------------

    reference_fields = [
        record.get("document_number"),
        record.get("invoice_number"),
        record.get("receipt_number"),
        record.get("po_number"),
        record.get("reference_numbers"),
    ]

    reference_present_flags = [
        present(value)
        for value in reference_fields
    ]

    reference_present_count = sum(reference_present_flags)

    reference_missing_count = (
        len(reference_fields) - reference_present_count
    )

    all_references_missing = int(
        reference_present_count == 0
    )

    document_number_missing = int(
        not present(record.get("document_number"))
    )

    any_secondary_reference_present = int(any([
        present(record.get("invoice_number")),
        present(record.get("receipt_number")),
        present(record.get("po_number")),
        present(record.get("reference_numbers")),
    ]))

    document_number_missing_but_secondary_present = int(
        document_number_missing == 1
        and any_secondary_reference_present == 1
    )

    # ---------------------------------------------------------
    # 7. Date consistency checks
    # ---------------------------------------------------------

    document_date = get_date(record.get("document_date"))
    due_date = get_date(record.get("due_date"))
    payment_date = get_date(record.get("payment_date"))

    document_date_present = int(pd.notna(document_date))
    due_date_present = int(pd.notna(due_date))
    payment_date_present = int(pd.notna(payment_date))

    due_before_document = int(
        pd.notna(due_date)
        and pd.notna(document_date)
        and due_date < document_date
    )

    payment_before_document = int(
        pd.notna(payment_date)
        and pd.notna(document_date)
        and payment_date < document_date
    )

    payment_after_due = int(
        pd.notna(payment_date)
        and pd.notna(due_date)
        and payment_date > due_date
    )

    payment_before_due = int(
        pd.notna(payment_date)
        and pd.notna(due_date)
        and payment_date < due_date
    )

    # ---------------------------------------------------------
    # 8. Currency consistency
    # ---------------------------------------------------------

    currency = record.get("document_currency")

    currency_text = (
        str(currency).strip().upper()
        if present(currency)
        else ""
    )

    line_currencies = [
        str(item.get("currency")).strip().upper()
        for item in line_items
        if present(item.get("currency"))
    ]

    currency_conflict_count = 0

    if currency_text:
        currency_conflict_count += sum(
            line_currency != currency_text
            for line_currency in line_currencies
        )

    elif line_currencies:
        # The document currency is unknown, so conflicts against it
        # cannot be established. Check whether line currencies disagree
        # with one another instead.
        first_currency = line_currencies[0]

        currency_conflict_count += sum(
            line_currency != first_currency
            for line_currency in line_currencies[1:]
        )

    currency_mismatch_flag = int(
        currency_conflict_count > 0
    )

    # ---------------------------------------------------------
    # 9. Tax details
    # ---------------------------------------------------------

    tax_rates = [
        str(item.get("tax_rate")).strip()
        for item in line_items
        if present(item.get("tax_rate"))
    ]

    line_tax_rate_count = len(set(tax_rates))

    # ---------------------------------------------------------
    # 9b. Tax summary table (added: lets the tax-breakdown model see
    #     the summary table, which earlier features ignored entirely)
    # ---------------------------------------------------------

    summary_taxes = [to_number(row.get("tax_amount")) for row in taxes
                     if isinstance(row, dict)]
    summary_bases = [to_number(row.get("taxable_base")) for row in taxes
                     if isinstance(row, dict)]

    tax_summary_sum = safe_sum(summary_taxes, require_all=True)
    tax_summary_difference = (
        tax_summary_sum - tax_amount
        if pd.notna(tax_summary_sum) and pd.notna(tax_amount)
        else np.nan
    )
    taxable_base_sum = safe_sum(summary_bases, require_all=True)
    reference_base = net_amount if pd.notna(net_amount) else line_amount_sum
    taxable_base_difference = (
        taxable_base_sum - reference_base
        if pd.notna(taxable_base_sum) and pd.notna(reference_base)
        else np.nan
    )

    # ---------------------------------------------------------
    # 9c. Tolerance-aware indicators (same tolerance as the rule engine)
    #     NaN means "could not be evaluated", never "no discrepancy".
    # ---------------------------------------------------------

    sum_tolerance = AMOUNT_TOLERANCE + 0.01 * len(line_items)

    def flag(difference, tolerance=sum_tolerance):
        if pd.isna(difference):
            return np.nan
        return float(abs(difference) > tolerance)

    def rate_fraction(value):
        import re
        match = re.search(r"\d+(?:\.\d+)?", str(value)) if value else None
        return float(match.group(0)) / 100 if match else np.nan

    expected_pairs = [
        (to_number(row.get("taxable_base")), rate_fraction(row.get("tax_rate")))
        for row in taxes if isinstance(row, dict)
    ]
    if expected_pairs and all(
        pd.notna(base) and pd.notna(rate) for base, rate in expected_pairs
    ):
        expected_tax_from_summary = float(
            sum(base * rate for base, rate in expected_pairs))
    else:
        expected_tax_from_summary = np.nan

    tax_rate_difference = (
        tax_amount - expected_tax_from_summary
        if pd.notna(tax_amount) and pd.notna(expected_tax_from_summary)
        else np.nan
    )

    # ---------------------------------------------------------
    # 10. Return model features
    # ---------------------------------------------------------

    return {
        # Invoice-level amounts
        "net_amount": net_amount,
        "tax_amount": tax_amount,
        "total_amount": total_amount,
        "calculated_total": calculated_total,
        "total_difference": total_difference,
        "absolute_total_difference": absolute_total_difference,

        # Line-item totals
        "line_item_count": len(line_items),
        "line_amount_sum": line_amount_sum,
        "line_tax_sum": line_tax_sum,
        "line_net_difference": line_net_difference,
        "absolute_line_net_difference": absolute_line_net_difference,
        "line_tax_difference": line_tax_difference,
        "absolute_line_tax_difference": absolute_line_tax_difference,

        # Line-item arithmetic
        "line_amount_mismatch_count": amount_mismatch_count,
        "line_amount_absolute_error_sum": amount_absolute_error_sum,
        "line_amount_checks_performed": line_amount_checks_performed,
        "missing_amount_comparison_count": missing_amount_comparison_count,

        # Missing line-item details
        "missing_line_amount_count": count_missing(line_amounts),
        "missing_line_tax_count": count_missing(line_taxes),
        "missing_quantity_count": count_missing(quantities),
        "missing_unit_price_count": count_missing(unit_prices),
        "missing_description_count": missing_description_count,
        "missing_line_tax_rate_count": missing_line_tax_rate_count,

        # Supplier details
        "supplier_missing_count": supplier_missing_count,
        "supplier_name_missing": supplier_name_missing,
        "supplier_address_missing": supplier_address_missing,
        "supplier_registration_missing": supplier_registration_missing,
        "supplier_phone_missing": supplier_phone_missing,
        "supplier_email_missing": supplier_email_missing,
        "supplier_website_missing": supplier_website_missing,
        "supplier_contact_missing_count": supplier_contact_missing_count,

        # Document references
        "reference_missing_count": reference_missing_count,
        "reference_present_count": reference_present_count,
        "all_references_missing": all_references_missing,
        "document_number_missing": document_number_missing,
        "any_secondary_reference_present": any_secondary_reference_present,
        "document_number_missing_but_secondary_present":
            document_number_missing_but_secondary_present,
        "document_number_present": present(record.get("document_number")),
        "invoice_number_present": present(record.get("invoice_number")),
        "receipt_number_present": present(record.get("receipt_number")),
        "po_number_present": present(record.get("po_number")),

        # Currency
        "document_currency_present": present(currency),
        "currency_is_usd": int(currency_text == "USD"),
        "currency_conflict_count": currency_conflict_count,
        "currency_mismatch_flag": currency_mismatch_flag,
        "line_currency_count": len(line_currencies),

        # Dates
        "document_date_present": document_date_present,
        "due_date_present": due_date_present,
        "payment_date_present": payment_date_present,
        "due_before_document": due_before_document,
        "payment_before_document": payment_before_document,
        "payment_after_due": payment_after_due,
        "payment_before_due": payment_before_due,

        # Tax details
        "tax_detail_count": len(taxes),
        "line_tax_rate_count": line_tax_rate_count,
        "tax_summary_sum": tax_summary_sum,
        "tax_summary_difference": tax_summary_difference,
        "absolute_tax_summary_difference": absolute_difference(
            tax_summary_sum, tax_amount),
        "taxable_base_difference": taxable_base_difference,
        "expected_tax_from_summary": expected_tax_from_summary,
        "tax_rate_difference": tax_rate_difference,
        "absolute_tax_rate_difference": (
            abs(tax_rate_difference) if pd.notna(tax_rate_difference) else np.nan),
        "line_arithmetic_flag": (
            float(amount_mismatch_count > 0)
            if line_amount_checks_performed > 0 else np.nan),
        "total_mismatch_flag": flag(total_difference),
        "line_net_mismatch_flag": flag(line_net_difference),
        "line_tax_mismatch_flag": flag(line_tax_difference),
        "tax_summary_mismatch_flag": flag(tax_summary_difference),
        "tax_rate_mismatch_flag": flag(tax_rate_difference),
        "absolute_taxable_base_difference": (
            abs(taxable_base_difference)
            if pd.notna(taxable_base_difference) else np.nan),
    }
