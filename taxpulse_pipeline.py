"""Single entry point that runs the full TaxPulse pipeline.

    PDF bytes -> parse -> features -> rule checks -> ML signals -> verdict

The UI and any batch script call ``analyze_invoice``; the PDF report is
built from the dictionary it returns.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from taxpulse_features import extract_invoice_features
from taxpulse_parser import parse_invoice_pdf
from taxpulse_report import create_pdf_report, predict_invoice
from taxpulse_rules import overall_verdict, run_rule_checks

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "taxpulse_models" / "taxpulse_gnb_models.joblib"
)


def load_bundle(path=DEFAULT_MODEL_PATH):
    """Load the trained bundle. Returns ``(bundle, error_message)``."""
    path = Path(path)
    if not path.exists():
        return None, f"Model bundle not found at {path.name}."
    try:
        import joblib
        bundle = joblib.load(path)
    except Exception as error:  # corrupt file, missing class, version clash
        return None, f"Model bundle could not be loaded: {error}"

    if not isinstance(bundle, dict) or not isinstance(bundle.get("models"), dict):
        return None, "Model bundle has an unexpected format."
    if not isinstance(bundle.get("model_columns"), dict):
        return None, "Model bundle has no 'model_columns' entry."
    return bundle, None


def _feature_frame(record):
    features = extract_invoice_features(record)
    frame = pd.DataFrame([features]) if isinstance(features, dict) else features.copy()
    if len(frame) != 1:
        raise ValueError("Feature extraction must produce exactly one row.")
    if frame.columns.duplicated().any():
        raise ValueError("Feature extraction produced duplicate columns.")
    return frame.replace([np.inf, -np.inf], np.nan)


def analyze_invoice(source, file_name=None, bundle=None, bundle_error=None,
                    build_report=True):
    """Run every stage and return one analysis dictionary."""
    record, text, warnings = parse_invoice_pdf(source, file_name=file_name)
    feature_row = _feature_frame(record)
    findings = run_rule_checks(record)

    ml_results, ml_error = None, bundle_error
    if bundle is not None:
        try:
            ml_results = predict_invoice(feature_row, bundle)
        except Exception as error:
            ml_error = f"Model scoring failed: {error}"

    level, headline, detail = overall_verdict(findings, warnings)
    ml_flagged = sum(1 for r in ml_results if r["flagged"]) if ml_results else 0
    if level == "clear" and ml_flagged:
        detail += (f" {ml_flagged} low-confidence pattern signal(s) were raised "
                   "by the ML models; see the Risk signals tab.")

    analysis = {
        "file_name": file_name or record.get("file_name") or "invoice.pdf",
        "generated_at": datetime.now().strftime("%d %b %Y, %H:%M"),
        "record": record,
        "extracted_text": text,
        "warnings": warnings,
        "feature_row": feature_row,
        "rule_findings": findings,
        "verdict": (level, headline, detail),
        "ml_results": ml_results,
        "ml_error": ml_error,
    }
    analysis["report_pdf"] = create_pdf_report(analysis) if build_report else None
    return analysis
