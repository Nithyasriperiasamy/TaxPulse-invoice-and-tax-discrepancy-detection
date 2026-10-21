# TaxPulse end-to-end evaluation
1000/1000 generated PDFs processed (seed 424242).

| category                         |   positives |   rules_accuracy |   rules_f1 |   rules_fp |   rules_fn |   ml_accuracy |   ml_f1 |   ml_fp |   ml_fn |
|:---------------------------------|------------:|-----------------:|-----------:|-----------:|-----------:|--------------:|--------:|--------:|--------:|
| tax_amount_mismatch              |         267 |                1 |          1 |          0 |          0 |             1 |       1 |       0 |       0 |
| invoice_total_mismatch           |         262 |                1 |          1 |          0 |          0 |             1 |       1 |       0 |       0 |
| incomplete_supplier_details      |         257 |                1 |          1 |          0 |          0 |             1 |       1 |       0 |       0 |
| line_item_arithmetic             |         244 |                1 |          1 |          0 |          0 |             1 |       1 |       0 |       0 |
| tax_breakdown_mismatch           |         240 |                1 |          1 |          0 |          0 |             1 |       1 |       0 |       0 |
| document_reference_inconsistency |         244 |                1 |          1 |          0 |          0 |             1 |       1 |       0 |       0 |

Invoice-level (any discrepancy): accuracy 1.0000, precision 1.0000, recall 1.0000

Field extraction accuracy:
- document_total_net: 1.0000
- document_total_tax: 1.0000
- document_total_amount: 1.0000
- invoice_number: 1.0000
- document_currency: 1.0000
- document_date: 1.0000
- supplier_name: 1.0000
- line item count: 1.0000