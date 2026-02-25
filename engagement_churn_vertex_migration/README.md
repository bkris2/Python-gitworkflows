# Engagement Propensity & Churn Templates (SageMaker ↔ Vertex AI)

This project provides reusable module templates for:

- AWS SageMaker Engagement Propensity
- AWS SageMaker Churn Prediction
- GCP Vertex AI Engagement Propensity (migration equivalent)
- GCP Vertex AI Churn Prediction (migration equivalent)

## Structure

- `aws_sagemaker/engagement_propensity_sagemaker_template.py`
- `aws_sagemaker/churn_prediction_sagemaker_template.py`
- `gcp_vertex_ai/engagement_propensity_vertex_template.py`
- `gcp_vertex_ai/churn_prediction_vertex_template.py`
- `configs/sagemaker_config.template.json`
- `configs/vertex_ai_config.template.json`
- `migration/aws_to_vertex_model_mapping.md`
- `migration/migration_runbook.md`

## Quick Start

1. Install dependencies (already present in repo `requirements.txt`):
   - `sagemaker`
   - `google-cloud-aiplatform`
   - `google-cloud-storage`

2. Copy and fill templates:
   - `configs/sagemaker_config.template.json`
   - `configs/vertex_ai_config.template.json`

3. Use SageMaker templates for current AWS training/deployment.

4. Use Vertex AI templates to migrate the same use cases with equivalent flow.

## Notes

- Templates are intentionally minimal and deployment-safe for adaptation.
- Keep feature engineering logic and schema aligned across both platforms.
- Prefer managing secrets via IAM roles / service accounts, not hardcoded keys.

## Model Parity Checklist

| Model | AWS SageMaker Template | Vertex AI Template | Target Column | Suggested Primary Metric |
|---|---|---|---|---|
| Engagement Propensity | `aws_sagemaker/engagement_propensity_sagemaker_template.py` | `gcp_vertex_ai/engagement_propensity_vertex_template.py` | `engaged` | ROC-AUC |
| Churn Prediction | `aws_sagemaker/churn_prediction_sagemaker_template.py` | `gcp_vertex_ai/churn_prediction_vertex_template.py` | `is_churned` | PR-AUC / F1 |

### Validation Before Cutover

- Use the same dataset snapshot and feature schema in both clouds.
- Compare score distributions and threshold behavior, not only single metrics.
- Validate online payload contract (field names, types, and ordering).
- Keep model version tags aligned across AWS and GCP releases.
