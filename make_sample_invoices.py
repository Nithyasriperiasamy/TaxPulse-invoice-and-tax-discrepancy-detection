"""Create demo invoices in sample_invoices/ (one clean, one with faults).

    python make_sample_invoices.py
"""
from pathlib import Path

import numpy as np

from synthetic import clean_record, inject_faults, write_invoice_pdf


def _find_faulty(want):
    """Deterministically find an invoice whose faults are exactly ``want``."""
    for seed in range(5000):
        rng = np.random.default_rng(seed)
        record, faults = inject_faults(clean_record(rng), rng, p=0.3)
        if {c for c, v in faults.items() if v} == want:
            return record
    raise RuntimeError("no matching invoice found")


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_invoices"
    out.mkdir(exist_ok=True)

    rng = np.random.default_rng(11)
    clean, _ = inject_faults(clean_record(rng), rng, p=0.0)
    write_invoice_pdf(clean, str(out / "clean_invoice.pdf"))

    faulty = _find_faulty({"line_item_arithmetic", "invoice_total_mismatch",
                           "incomplete_supplier_details"})
    write_invoice_pdf(faulty, str(out / "faulty_invoice.pdf"))
    print("Created", out)
