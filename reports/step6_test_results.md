# Held-out test evaluation

The classifier, feature count, bit width, and threshold were frozen in
`reports/freeze.json` before this file was written. This score did not
change them.

Model: logistic regression, 16 features, threshold 0.50.
Fixed point: 12 bits (6 input fractional bits, 10 weight fractional bits).
Test rows: 314 (21 failures, 293 passes).

## Prediction metrics

| Decision | Recall | False-positive rate | Precision | Escapes | False alarms | Screening cost | Agreement with float |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Floating point | 0.5714 | 0.2423 | 0.1446 | 9 | 71 | 161 | 1.0000 |
| 12-bit | 0.5714 | 0.2423 | 0.1446 | 9 | 71 | 161 | 1.0000 |

Floating-point average precision (PR-AUC) 0.1567, ROC-AUC 0.7247, accuracy 0.7452.
Confusion, floating point: TP 12, FN 9, FP 71, TN 222.
Parallel RTL on these test vectors: passed.
Sequential MAC RTL on these test vectors: passed.

Screening cost = 10 × escaped failures + 1 × false alarms. The weights are the same illustration used on the validation set. They are not fab prices.

## Limitations

- The test split has 21 failures. A recall difference of one wafer moves the rate by about 4.8%.
- Feature count, bit width, and threshold were chosen on the validation split. This test score can look worse than the validation score, and it must not be used to pick a new one.
- Class-weighted logistic scores are not calibrated probabilities.
- Screening cost uses illustrative weights of 10 per escaped failure and 1 per false alarm. It is not a fab cost.
- Sensor names are anonymous. A selected column is not a demonstrated physical cause.
- The RTL sees train-median-imputed, standardized, selected features. Raw-sensor preprocessing is outside the chip.
- Python/RTL agreement on this file checks the frozen integer decision. It is not a new accuracy claim beyond the counts above.
- Nangate45 area and timing live in the hardware reports. They have no wire-load model, no clock tree, and no power number.
