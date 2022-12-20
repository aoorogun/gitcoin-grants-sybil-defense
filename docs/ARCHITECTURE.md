# Architecture

```text
on-chain contributions + trust attestations -> point-in-time donor features
                   |                              |
             anomaly model                  supervised risk model
                   \                              /
                       human review + appeal
                                |
           eligible/downweighted contribution matrix -> QF calculator
                                |
             shadow comparison, reconciliation, signed settlement artifact
```

On AWS, an indexed event stream lands in versioned S3. Glue/SageMaker Processing creates point-in-time donor features. SageMaker Pipelines trains and evaluates; Model Registry requires approval. Batch Transform scores the complete round before settlement. A Lambda/API service handles analyst what-if scenarios. CloudWatch monitors data freshness, drift, review load and allocation reconciliation. Every settlement stores source block range, model, thresholds, appeals and output checksum.

