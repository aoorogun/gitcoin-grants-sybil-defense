# Model card

## Intended use

Prioritise donor accounts for human Sybil review and estimate how risk policies affect quadratic-funding allocations. A flag is not proof of shared control or malicious intent.

## Data

The GR14 JSON contains real published aggregates. All wallet-level records, labels, fingerprints and contributions are simulated and deliberately non-linkable. Production labels should combine trained human review, transparent heuristics and adjudicated appeals; naïvely treating previous model flags as truth creates feedback loops.

## Evaluation and governance

Use chronological train/validation/test splits. Report ROC-AUC, average precision, Brier score, precision, recall and false-positive rate. Matching changes are reported separately because classification metrics do not express allocation harm. Block recommendations remain pending appeal. Threshold or feature changes require shadow settlement and governance approval.

## Risks

New wallets, privacy tools, shared infrastructure and low-value participation can resemble Sybil behaviour. Identity/trust scores can exclude legitimate users. Coordinated attackers adapt. The simulator simplifies graphs and does not validate production performance.

