"""End-to-end accuracy check on generated PDFs.

    python evaluate.py [--n 400] [--seed 2025]

For each synthetic invoice with known faults it renders a real PDF, runs the
whole pipeline (parser -> features -> rules -> ML) and compares the result
with the faults that were injected. Use a seed that differs from training.
Writes evaluation_report.csv and evaluation_report.md.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from synthetic import CATEGORIES, invoice_pdf_bytes, make_dataset
from taxpulse_pipeline import analyze_invoice, load_bundle
from taxpulse_rules import category_status

HERE = Path(__file__).resolve().parent
# Rule statuses that count as "this category has a discrepancy".
POSITIVE = {
    "tax_amount_mismatch": {"fail"},
    "invoice_total_mismatch": {"fail"},
    "line_item_arithmetic": {"fail"},
    "tax_breakdown_mismatch": {"fail"},
    "incomplete_supplier_details": {"fail", "warn"},
    "document_reference_inconsistency": {"fail"},
}
FIELDS = [
    ("document_total_net", 0.005), ("document_total_tax", 0.005),
    ("document_total_amount", 0.005), ("invoice_number", None),
    ("document_currency", None), ("document_date", None),
    ("supplier_name", None),
]


def metrics(y, p):
    tp = sum(a and b for a, b in zip(y, p))
    tn = sum((not a) and (not b) for a, b in zip(y, p))
    fp = sum((not a) and b for a, b in zip(y, p))
    fn = sum(a and (not b) for a, b in zip(y, p))
    n = len(y)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n=n, accuracy=(tp + tn) / n, precision=prec, recall=rec,
                f1=f1, tp=tp, fp=fp, tn=tn, fn=fn)


def main(n, seed):
    bundle, error = load_bundle()
    data = make_dataset(n, seed)
    truth = {c: [] for c in CATEGORIES}
    rules = {c: [] for c in CATEGORIES}
    ml = {c: [] for c in CATEGORIES}
    field_ok = {f: 0 for f, _ in FIELDS}
    item_ok = parse_fail = 0
    misses = []

    for index, (record, faults) in enumerate(data):
        try:
            a = analyze_invoice(invoice_pdf_bytes(record), f"eval_{index}.pdf",
                                bundle, error, build_report=False)
        except Exception as exc:  # a crash counts as a failure
            parse_fail += 1
            misses.append((index, "pipeline crash", str(exc)))
            continue
        status = category_status(a["rule_findings"])
        by_cat = {r["category"]: r for r in (a["ml_results"] or [])}
        parsed = a["record"]
        for c in CATEGORIES:
            truth[c].append(bool(faults[c]))
            rules[c].append(status.get(c) in POSITIVE[c])
            ml[c].append(bool(by_cat[c]["flagged"]) if c in by_cat else False)
            if truth[c][-1] != rules[c][-1]:
                misses.append((index, c, f"truth={truth[c][-1]} rules={rules[c][-1]}"))
        for f, tol in FIELDS:
            want, got = record.get(f), parsed.get(f)
            ok = (abs(want - got) <= tol * max(1, abs(want))) if (
                tol is not None and want is not None and got is not None) else want == got
            field_ok[f] += ok
        item_ok += len(parsed["line_items"]) == len(record["line_items"])

    done = n - parse_fail
    rows = []
    for c in CATEGORIES:
        r, m = metrics(truth[c], rules[c]), metrics(truth[c], ml[c])
        rows.append({"category": c, "positives": sum(truth[c]),
                     "rules_accuracy": r["accuracy"], "rules_f1": r["f1"],
                     "rules_fp": r["fp"], "rules_fn": r["fn"],
                     "ml_accuracy": m["accuracy"], "ml_f1": m["f1"],
                     "ml_fp": m["fp"], "ml_fn": m["fn"]})
    table = pd.DataFrame(rows)
    # Overall: invoice-level "has any discrepancy" decision
    any_truth = [any(truth[c][i] for c in CATEGORIES) for i in range(done)]
    any_rules = [any(rules[c][i] for c in CATEGORIES) for i in range(done)]
    overall = metrics(any_truth, any_rules)

    lines = [f"# TaxPulse end-to-end evaluation",
             f"{done}/{n} generated PDFs processed (seed {seed}).", "",
             table.round(4).to_markdown(index=False), "",
             f"Invoice-level (any discrepancy): accuracy {overall['accuracy']:.4f}, "
             f"precision {overall['precision']:.4f}, recall {overall['recall']:.4f}", "",
             "Field extraction accuracy:"]
    for f, _ in FIELDS:
        lines.append(f"- {f}: {field_ok[f] / done:.4f}")
    lines.append(f"- line item count: {item_ok / done:.4f}")
    if misses:
        lines += ["", f"First rule disagreements ({len(misses)} total):"]
        lines += [f"- invoice {i}: {c} {d}" for i, c, d in misses[:15]]
    report = "\n".join(lines)
    print(report)
    table.to_csv(HERE / "evaluation_report.csv", index=False)
    (HERE / "evaluation_report.md").write_text(report, encoding="utf-8")
    return 0 if (overall["accuracy"] >= 0.99 and not parse_fail) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    sys.exit(main(args.n, args.seed))
