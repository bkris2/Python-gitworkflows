# AWS SageMaker to Vertex AI Mapping (Engagement & Churn)

## Module Mapping

| AWS SageMaker Template | Vertex AI Equivalent |
|---|---|
| `EngagementPropensitySageMakerTemplate` | `EngagementPropensityVertexTemplate` |
| `ChurnPredictionSageMakerTemplate` | `ChurnPredictionVertexTemplate` |

## Concept Mapping

| SageMaker Concept | Vertex AI Equivalent |
|---|---|
| `SKLearn Estimator` | `CustomContainerTrainingJob` |
| `estimator.fit()` | `training_job.run()` |
| `estimator.deploy()` | `model.deploy(endpoint=...)` |
| S3 training channels | GCS URIs / staged artifacts |
| IAM role ARN | GCP service account |

## Target Schema Guidance

- Engagement target column: `engaged`
- Churn target column: `is_churned`
- Keep feature names and preprocessing logic identical across clouds.
- Keep model thresholds/version tags aligned for A/B parity checks.
