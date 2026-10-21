"""Compatibility and regression tests.   Run:  python test_pipeline.py
(also works with pytest).  Exit code is non-zero if anything fails."""
import copy
import sys
import traceback

import numpy as np

from synthetic import clean_record, inject_faults, invoice_pdf_bytes, make_dataset
from taxpulse_features import extract_invoice_features
from taxpulse_model import ManualGaussianNB
from taxpulse_pipeline import analyze_invoice, load_bundle
from taxpulse_rules import run_rule_checks

BUNDLE, BUNDLE_ERROR = load_bundle()


def _clean():
    rng = np.random.default_rng(5)
    return inject_faults(clean_record(rng), rng, p=0.0)[0]


def _run(record, **kw):
    return analyze_invoice(invoice_pdf_bytes(record), "t.pdf", BUNDLE, BUNDLE_ERROR, **kw)


def test_bundle_loads_and_matches_features():
    assert BUNDLE is not None, BUNDLE_ERROR
    feats = extract_invoice_features(_clean())
    for cat, cols in BUNDLE["model_columns"].items():
        missing = [c for c in cols if c not in feats]
        assert not missing, f"{cat}: feature(s) not produced by extractor: {missing}"
    assert set(BUNDLE["models"]) == set(BUNDLE["model_columns"]) == set(BUNDLE["categories"])


def test_manual_nb_matches_sklearn():
    from sklearn.naive_bayes import GaussianNB
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (200, 3)), rng.normal(2, 1.5, (200, 3))])
    y = np.array([0] * 200 + [1] * 200)
    mine = ManualGaussianNB(var_smoothing=1e-9).fit(X, y)
    ref = GaussianNB(var_smoothing=1e-9).fit(X, y)
    assert (mine.predict(X) == ref.predict(X)).mean() > 0.995


def test_clean_invoice_has_no_failures():
    a = _run(_clean())
    assert a["verdict"][0] in ("clear", "attention")
    assert not [f for f in a["rule_findings"] if f["status"] == "fail"]
    assert not any(r["flagged"] for r in a["ml_results"])


def test_each_injected_fault_is_caught():
    from taxpulse_rules import category_status
    positive = {"incomplete_supplier_details": {"fail", "warn"}}
    wrong = 0
    for rec, faults in make_dataset(200, seed=77, p=0.3):
        status = category_status(run_rule_checks(rec))
        for cat, present in faults.items():
            hit = status.get(cat) in positive.get(cat, {"fail"})
            wrong += hit != bool(present)
    assert wrong == 0, f"{wrong} rule/ground-truth disagreements"


def test_rounding_noise_is_not_a_fault():
    rec = _clean()
    rec = copy.deepcopy(rec)
    rec["line_items"][0]["tax_amount"] = round(rec["line_items"][0]["tax_amount"] + 0.01, 2)
    a = _run(rec, build_report=False)
    assert not [f for f in a["rule_findings"] if f["status"] == "fail"]


def test_wrong_total_is_flagged_by_rules_and_ml():
    rec = copy.deepcopy(_clean())
    rec["document_total_amount"] += 50
    a = _run(rec, build_report=False)
    assert a["verdict"][0] == "review"
    assert next(r for r in a["ml_results"] if r["category"] == "invoice_total_mismatch")["flagged"]


def test_missing_data_is_not_evaluable_not_clean():
    from taxpulse_report import predict_invoice
    import pandas as pd
    frame = pd.DataFrame([{c: np.nan for c in extract_invoice_features(_clean())}])
    results = predict_invoice(frame, BUNDLE)
    unevaluable = {r["category"] for r in results if not r["evaluable"]}
    assert "invoice_total_mismatch" in unevaluable and "line_item_arithmetic" in unevaluable


def test_works_without_model_bundle():
    a = analyze_invoice(invoice_pdf_bytes(_clean()), "t.pdf", None, "no bundle")
    assert a["ml_results"] is None and a["report_pdf"][:4] == b"%PDF"


def test_report_is_a_pdf():
    a = _run(_clean())
    assert a["report_pdf"][:4] == b"%PDF" and len(a["report_pdf"]) > 3000


def test_corrupt_pdf_gives_clear_error():
    try:
        analyze_invoice(b"not a pdf", "bad.pdf")
    except ValueError as e:
        assert "PDF" in str(e)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    print(f"\n{'ALL PASSED' if not failed else str(failed) + ' FAILED'}")
    sys.exit(1 if failed else 0)
