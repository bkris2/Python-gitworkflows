# Migration Runbook: SageMaker → Vertex AI

## 1) Prepare Baseline on AWS

- Freeze feature engineering code for engagement and churn models.
- Record training dataset snapshot and schema version.
- Record baseline metrics: AUC, F1, precision/recall, calibration.

## 2) Prepare GCP Environment

- Enable Vertex AI APIs.
- Create service account with Vertex AI + GCS access.
- Configure staging bucket and region.

## 3) Port Training Configuration

- Translate SageMaker hyperparameters to Vertex train args.
- Keep target columns unchanged:
  - Engagement: `engaged`
  - Churn: `is_churned`

## 4) Train Vertex Equivalents

- Start with same feature set and objective.
- Run one canary training job per model.
- Register models with version labels:
  - `engagement-propensity-v1`
  - `churn-prediction-v1`

## 5) Validate Parity

- Compare AWS vs GCP metrics on identical validation split.
- Compare score distribution and decision thresholds.
- Validate online inference schema compatibility.

## 6) Cutover Strategy

- Deploy Vertex endpoint with low traffic first.
- Monitor drift, latency, and error rate.
- Increase traffic gradually after parity confidence.
