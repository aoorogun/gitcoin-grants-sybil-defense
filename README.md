# Gitcoin Grants Sybil Defense & Quadratic Funding Analytics

An end-to-end 2022-era reference project for detecting coordinated contributor behaviour, protecting quadratic-funding matching pools, and simulating policy changes before round settlement.

**Data statement:** Gitcoin’s published GR14 aggregate statistics are stored as a real public benchmark. Row-level wallets, contributions and labels are privacy-safe simulations calibrated to that public context; they are not Gitcoin production exports or confidential Sybil determinations.

## What the project demonstrates

- Supervised ML: donor-level Sybil-risk classification
- Unsupervised ML: Isolation Forest anomaly detection
- Graph-inspired features: shared-device group size, grant overlap, contribution bursts and concentration
- Funding mechanism: quadratic funding `(sum(sqrt(contribution)))²`
- Dynamic reforecasting: matching allocations before/after risk policies
- Human-in-the-loop: high-risk flags form a review queue rather than automatic accusations
- Production ML: chronological evaluation, model artifacts, API, drift monitoring, CI and AWS design
- Deep learning: optional PyTorch donor-embedding classifier

## Historical context

Gitcoin’s GR14 governance brief reported 44,886 contributors, 623,400 contributions, 16,073 flagged contributors, 167,988 flagged contributions and approximately $879k in mitigated “fraud tax”. This project stores those aggregate values in `data/public/gr14_public_benchmark.json` and uses them only as transparent calibration context.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

gitcoin-integrity generate
gitcoin-integrity train
gitcoin-integrity allocate
gitcoin-integrity scenario --file configs/scenario_stricter_policy.json
gitcoin-integrity monitor
python -m unittest discover -s tests -v
```

## Repository map

```text
data/public/                 cited GR14 aggregate benchmark
data/simulated/              privacy-safe contributor and contribution data
src/gitcoin_integrity/       features, models, QF allocation, scenarios and API
artifacts/                   model, anomaly detector and metadata
reports/                     evaluation, allocations and scenario impact
docs/                        model card, architecture, runbook and interview guide
infra/                       AWS baseline
```

## Responsible interpretation

Risk scores indicate behaviour requiring review. They are not proof that a person controls multiple identities. False positives can remove legitimate pluralistic participation and distort matching. Threshold changes therefore require human review, appeal, shadow calculations and signed governance approval.

