# Verified evaluation

The GR14 benchmark is a real published aggregate. All donor-level data and labels are simulated.

| Measure | Result |
|---|---:|
| Selected model | Logistic regression |
| Test ROC-AUC | 0.719 |
| Test average precision | 0.466 |
| Brier score | 0.066 |
| Review-threshold precision | 0.778 |
| Review-threshold recall | 0.538 |
| False-positive rate | 1.59% |

The default policy allows 93.5% of evaluated test donors, queues 2.9% for human review and marks 3.6% as block-pending-appeal. Under the stricter scenario, 115 simulated donors enter review and 1,591 are blocked across the full round. The largest grant allocation change is $538.49 and the $1m pool reconciles to the cent.

Drift is critical because contribution activity accumulates through the simulated round; a production monitor should compare matched maturity windows rather than treating this as an automatic model failure. Three tests, linting, compilation and the complete generate/train/allocate/scenario/monitor workflow passed.

