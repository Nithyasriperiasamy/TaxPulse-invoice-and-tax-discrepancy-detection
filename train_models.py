"""Train the six per-category Gaussian Naive Bayes models.

    python train_models.py [--rows 6000] [--seed 42]

Generates synthetic invoices (synthetic.py) with known, isolated faults, turns them
into features with ``extract_invoice_features`` and writes a bundle to
taxpulse_models/taxpulse_gnb_models.joblib in the format the app expects:

    {"models", "categories", "model_columns", "thresholds", "metrics"}

If you already have a bundle trained on your own data, you do not need to
run this: put your file in the same place and the app will use it.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from taxpulse_features import extract_invoice_features
from synthetic import CATEGORIES as SYN_CATEGORIES, make_dataset
from taxpulse_model import ManualGaussianNB

OUT = Path(__file__).resolve().parent / "taxpulse_models"

CATEGORIES = SYN_CATEGORIES

MODEL_COLUMNS = {
    "tax_amount_mismatch": [
        "line_tax_mismatch_flag", "tax_rate_mismatch_flag",
        "absolute_line_tax_difference", "absolute_tax_rate_difference"],
    "invoice_total_mismatch": [
        "total_mismatch_flag", "absolute_total_difference"],
    "incomplete_supplier_details": [
        "supplier_address_missing", "supplier_phone_missing",
        "supplier_email_missing", "supplier_registration_missing",
        "supplier_name_missing", "supplier_missing_count",
        "supplier_contact_missing_count"],
    "line_item_arithmetic": [
        "line_arithmetic_flag", "line_net_mismatch_flag",
        "line_amount_absolute_error_sum", "absolute_line_net_difference"],
    "tax_breakdown_mismatch": [
        "tax_summary_mismatch_flag", "absolute_tax_summary_difference",
        "absolute_taxable_base_difference"],
    "document_reference_inconsistency": [
        "document_number_missing", "invoice_number_present",
        "reference_missing_count", "po_number_present",
        "all_references_missing"],
}


def best_threshold(y, p):
    """Threshold with the best validation F1 (middle one when several tie)."""
    grid = np.arange(0.05, 0.96, 0.05)
    scores = np.array([f1_score(y, p >= t, zero_division=0) for t in grid])
    tied = grid[scores >= scores.max() - 1e-9]
    return float(np.median(tied))


def main(rows, seed):
    data = make_dataset(rows, seed)
    feats = [extract_invoice_features(rec) for rec, _ in data]
    labs = [faults for _, faults in data]
    X = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)
    Y = pd.DataFrame(labs)

    idx_train, idx_rest = train_test_split(np.arange(rows), test_size=0.3, random_state=seed)
    idx_val, idx_test = train_test_split(idx_rest, test_size=0.5, random_state=seed)

    models, thresholds, metrics = {}, {}, []
    for cat in CATEGORIES:
        cols = MODEL_COLUMNS[cat]
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", ManualGaussianNB()),
        ]).fit(X.loc[idx_train, cols], Y.loc[idx_train, cat])

        p_val = pipe.predict_proba(X.loc[idx_val, cols])[:, 1]
        thr = best_threshold(Y.loc[idx_val, cat], p_val)
        p_test = pipe.predict_proba(X.loc[idx_test, cols])[:, 1]
        y_test = Y.loc[idx_test, cat]
        models[cat], thresholds[cat] = pipe, thr
        metrics.append({
            "category": cat, "threshold": thr,
            "precision": precision_score(y_test, p_test >= thr, zero_division=0),
            "recall": recall_score(y_test, p_test >= thr, zero_division=0),
            "f1": f1_score(y_test, p_test >= thr, zero_division=0),
            "roc_auc": roc_auc_score(y_test, p_test),
        })

    OUT.mkdir(exist_ok=True)
    bundle = {"models": models, "categories": CATEGORIES,
              "model_columns": MODEL_COLUMNS, "thresholds": thresholds,
              "metrics": metrics, "trained_on": "synthetic"}
    joblib.dump(bundle, OUT / "taxpulse_gnb_models.joblib")
    print(pd.DataFrame(metrics).round(3).to_string(index=False))
    print("Saved", OUT / "taxpulse_gnb_models.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.rows, a.seed)
