# TaxPulse

Upload a text-based invoice PDF. TaxPulse extracts the fields, recalculates the
amounts, scores pattern-based risk signals and builds a downloadable PDF report.

## Run
```
pip install -r requirements.txt
python train_models.py        # only if taxpulse_models/ is missing or you want to retrain
streamlit run app.py
```

## Verify (no Streamlit needed)
```
python test_pipeline.py       # 10 compatibility / regression tests
python evaluate.py            # end-to-end accuracy on generated PDFs
```

## Flow
```
PDF bytes
  -> taxpulse_parser     fields, line items, tax summary rows, warnings
  -> taxpulse_features   numeric features incl. tolerance-aware flags (NaN = not evaluable)
  -> taxpulse_rules      exact checks  (authoritative verdict)
  -> taxpulse_report.predict_invoice   Naive Bayes risk signals (secondary)
  -> verdict + PDF report      all via taxpulse_pipeline.analyze_invoice()
app.py is a thin Streamlit UI over that single function.
```

| File | Role |
|---|---|
| app.py | Streamlit interface, downloads (PDF, CSV, JSON) |
| taxpulse_pipeline.py | One-call orchestration, model loading |
| taxpulse_parser.py | PDF text to structured record |
| taxpulse_features.py | Record to model features |
| taxpulse_rules.py | Exact arithmetic and completeness checks |
| taxpulse_model.py | `ManualGaussianNB` (matches scikit-learn's GaussianNB) |
| taxpulse_report.py | Model scoring, explanations, PDF report |
| synthetic.py | Invoice generator with isolated, known faults (+ PDF writer) |
| train_models.py | Trains the six models into taxpulse_models/ |
| evaluate.py | End-to-end accuracy test, writes evaluation_report.* |
| test_pipeline.py | Tests |

## What the accuracy numbers mean
`evaluate.py` renders real PDFs from synthetic invoices, runs the full pipeline and
compares with the injected faults. Results are ~100% because the synthetic invoices
use the template the parser was built around and faults are cleanly separable.
It proves the stages are consistent with each other. It does **not** prove accuracy
on arbitrary real-world invoices: scanned PDFs need OCR, and unusual layouts may need
parser adjustments. Validate on a sample of your own invoices.

* Rule checks decide the verdict. ML signals never override them.
* A missing/failed model bundle falls back to rules only.
* To use your own trained bundle, place it at `taxpulse_models/taxpulse_gnb_models.joblib`
  with keys `models` and `model_columns` (optional: `categories`, `thresholds`).
