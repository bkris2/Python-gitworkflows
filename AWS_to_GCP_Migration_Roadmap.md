# AWS SageMaker to GCP Vertex AI Migration Roadmap

## Table of Contents
1. [Phase 1: Inventory Collection](#phase-1-inventory-collection)
2. [Phase 2: Service & Module Mapping](#phase-2-service--module-mapping)
3. [Phase 3: Step-by-Step Migration](#phase-3-step-by-step-migration)
4. [Phase 4: Testing Scenarios](#phase-4-testing-scenarios)
5. [Phase 5: Deployment](#phase-5-deployment)

---

## Phase 1: Inventory Collection

### 1.1 AWS SageMaker Resources Inventory

Create an inventory document capturing:

#### Training Assets
- [ ] Training job names and configurations
- [ ] Training script locations (S3 paths)
- [ ] Datasets used (S3 bucket paths, sizes)
- [ ] Instance types and counts
- [ ] Hyperparameters configurations
- [ ] Training job history and performance metrics
- [ ] Role ARNs and IAM permissions

#### Models & Endpoints
- [ ] Model artifacts location and size
- [ ] Endpoint configurations
- [ ] Endpoint deployment instances
- [ ] Load balancing settings
- [ ] Traffic distribution rules
- [ ] Model versioning strategy

#### Data & Storage
- [ ] S3 bucket structures
- [ ] Data preprocessing pipelines
- [ ] Feature store implementations
- [ ] Data validation rules

#### Integrated Services
- [ ] IAM roles and policies
- [ ] KMS encryption keys used
- [ ] CloudWatch monitoring rules
- [ ] SNS/SQS integrations
- [ ] Lambda functions linked to SageMaker

#### Code & Scripts
- [ ] Python scripts/libraries used
- [ ] Framework versions (TensorFlow, PyTorch, etc.)
- [ ] Custom training containers
- [ ] Preprocessing/postprocessing code

**Template CSV for Inventory:**
```
ResourceType,ResourceName,Config,Location,Owner,Dependencies,Notes
TrainingJob,linear-regression-job,instance-type:ml.m5.xlarge,s3://bucket/path,TeamA,SageMaker,
Model,model-v1,framework:scikit-learn,s3://bucket/models/,TeamA,TrainingJob,
Endpoint,prod-endpoint,instance:ml.m5.large,us-east-1,TeamA,Model,Live
```

---

## Phase 2: Service & Module Mapping

### 2.1 AWS to GCP Service Mapping

| AWS SageMaker | GCP Vertex AI | Migration Complexity |
|---|---|---|
| Training Jobs | Training Pipelines / CustomTrainingJob | Medium |
| Endpoints | Online Prediction Endpoints | Medium |
| Batch Transform | Batch Prediction | Low |
| Notebook Instances | Vertex AI Workbench | Low |
| Feature Store | Vertex Feature Store | High |
| Pipelines | Vertex AI Pipelines | Medium |
| Model Registry | Model Registry | Low |
| AutoML | Vertex AutoML | Low |
| Ground Truth | Data Labeling | Low |

### 2.2 Python Module & Configuration Mapping

#### SageMaker SDK Equivalents

**SageMaker Module** → **Vertex AI Module**

```python
# Training
import sagemaker                      → from google.cloud import aiplatform
from sagemaker.estimator import Estimator → aiplatform.training.CustomTrainingJob

# Endpoints
from sagemaker.model import Model    → aiplatform.Model
from sagemaker.predictor import Predictor → aiplatform.Endpoint

# Monitoring
from sagemaker.model_monitor import ModelMonitor → aiplatform.monitoring

# Feature Store
from sagemaker.feature_store.feature_group import FeatureGroup → aiplatform.FeatureStore
```

### 2.3 Configuration File Mapping

**sagemaker_config.json**
```json
{
  "role": "arn:aws:iam::123456789:role/SageMaker",
  "instance_type": "ml.m5.xlarge",
  "instance_count": 1,
  "region": "us-east-1",
  "bucket": "sagemaker-bucket"
}
```

**vertex_ai_config.json** (GCP Equivalent)
```json
{
  "project_id": "my-gcp-project",
  "region": "us-central1",
  "service_account": "vertex-ai-sa@my-gcp-project.iam.gserviceaccount.com",
  "machine_type": "n1-standard-4",
  "staging_bucket": "vertex-ai-staging-bucket"
}
```

### 2.4 Environment Variables Mapping

| SageMaker | Vertex AI |
|---|---|
| AWS_REGION | GOOGLE_CLOUD_PROJECT |
| SAGEMAKER_ROLE_ARN | GOOGLE_APPLICATION_CREDENTIALS |
| SAGEMAKER_SESSION_REGION | GOOGLE_CLOUD_REGION |
| S3_BUCKET | GCS_BUCKET |

---

## Phase 3: Step-by-Step Migration

### 3.1 Module Migration Sequence

#### Step 1: Setup GCP Environment
```bash
# Install GCP SDK
pip install google-cloud-aiplatform

# Authenticate
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

#### Step 2: Migrate Data to Cloud Storage
```python
# Migrate S3 data to GCS
from google.cloud import storage
import boto3

def migrate_s3_to_gcs(s3_bucket, gcs_bucket):
    s3 = boto3.client('s3')
    gcs = storage.Client()
    
    # List S3 objects
    response = s3.list_objects_v2(Bucket=s3_bucket)
    
    for obj in response.get('Contents', []):
        key = obj['Key']
        # Download from S3
        s3.download_file(s3_bucket, key, f'/tmp/{key}')
        # Upload to GCS
        bucket = gcs.bucket(gcs_bucket)
        blob = bucket.blob(key)
        blob.upload_from_filename(f'/tmp/{key}')
```

#### Step 3: Migrate Training Code
```python
# Before (SageMaker)
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri="382416733822.dkr.ecr.us-east-1.amazonaws.com/image:latest",
    role="arn:aws:iam::123456789:role/SageMaker",
    instance_count=1,
    instance_type="ml.m5.xlarge"
)

# After (Vertex AI)
from google.cloud import aiplatform

job = aiplatform.CustomTrainingJob(
    display_name="training-job",
    script_path="train.py",
    container_uri="gcr.io/my-project/image:latest",
    requirements=['tensorflow', 'scikit-learn']
)

job.run(
    machine_type="n1-standard-4",
    replica_count=1,
    gcs_output_dir="gs://bucket/output"
)
```

#### Step 4: Migrate Model Registry & Versioning
```python
# Register model in Vertex AI
model = aiplatform.Model.upload(
    display_name="linear-regression-model",
    artifact_uri="gs://bucket/model.pkl",
    serving_container_image_uri="gcr.io/cloud-aiplatform/prediction-py:latest"
)

# Get model version
print(f"Model ID: {model.resource_name}")
```

#### Step 5: Migrate Endpoints
```python
# Before (SageMaker)
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="prod-endpoint"
)

# After (Vertex AI)
endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3
)

# Make predictions
response = endpoint.predict(instances=[[1.0, 2.0, 3.0]])
```

#### Step 6: Migrate Monitoring & Logging
```python
# Enable Vertex AI Model Monitoring
from google.cloud.aiplatform import monitoring

model_deployment_monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name="monitoring-job",
    project="my-project",
    location="us-central1",
    objective_config=aiplatform.monitoring.ObjectiveConfig(
        deployed_model_id=endpoint.deployed_model_display_name,
        training_dataset=aiplatform.monitoring.TrainingDataset(dataset_id="dataset-id")
    )
)
```

### 3.2 Code Refactoring Example

**Original SageMaker Code (train.py)**
```python
import os
import sys
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

def train():
    # Read from S3
    training_data = np.load('/opt/ml/input/data/training/data.npy')
    labels = np.load('/opt/ml/input/data/training/labels.npy')
    
    # Train model
    model = LinearRegression()
    model.fit(training_data, labels)
    
    # Save model
    with open('/opt/ml/model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    train()
```

**Refactored for Vertex AI (train.py)**
```python
import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from google.cloud import storage

def train():
    # Configuration
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
    bucket_name = os.environ.get('BUCKET_NAME')
    
    # Read from GCS
    gcs_client = storage.Client(project=project_id)
    bucket = gcs_client.bucket(bucket_name)
    
    training_data = np.load(
        bucket.blob('training/data.npy').download_as_string()
    )
    labels = np.load(
        bucket.blob('training/labels.npy').download_as_string()
    )
    
    # Train model
    model = LinearRegression()
    model.fit(training_data, labels)
    
    # Save to GCS
    model_blob = bucket.blob('models/model.pkl')
    model_blob.upload_from_string(pickle.dumps(model))
    
    print(f"Model saved to gs://{bucket_name}/models/model.pkl")

if __name__ == '__main__':
    train()
```

---

## Phase 4: Testing Scenarios

### 4.1 Pre-Migration Testing

#### Unit Tests
```python
import pytest
from sklearn.linear_model import LinearRegression
from model_utils import load_model, preprocess_data

def test_model_training():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    model = LinearRegression()
    model.fit(X, y)
    assert model.score(X, y) > 0.5

def test_data_preprocessing():
    raw_data = np.array([[1, 2], [3, 4]])
    processed = preprocess_data(raw_data)
    assert processed.shape == raw_data.shape
    assert not np.isnan(processed).any()

def test_model_serialization():
    model = LinearRegression()
    model.fit(np.array([[1, 2]]), np.array([1]))
    
    import pickle
    serialized = pickle.dumps(model)
    deserialized = pickle.loads(serialized)
    
    assert type(deserialized) == LinearRegression
```

### 4.2 Migration Testing Checklist

- [ ] **Data Integrity Tests**
  - [ ] Source and destination data match
  - [ ] Data types preserved
  - [ ] No data loss during transfer
  - [ ] Data sampling validation

- [ ] **Model Performance Tests**
  - [ ] Accuracy matches within 1-2%
  - [ ] Inference latency acceptable
  - [ ] Throughput meets requirements
  - [ ] Memory usage within limits

- [ ] **Integration Tests**
  - [ ] Endpoint connectivity verified
  - [ ] API responses match expected format
  - [ ] Authentication working
  - [ ] Monitoring/logging active

- [ ] **Load Testing**
  - [ ] Test with 10x average load
  - [ ] Test with 100x average load
  - [ ] Measure autoscaling response
  - [ ] Check cost implications

### 4.3 A/B Testing Strategy

```python
# Route traffic to both endpoints during migration
# Collect metrics and compare results

def route_prediction_request(input_data):
    gcp_result = call_vertex_endpoint(input_data)
    aws_result = call_sagemaker_endpoint(input_data)
    
    # Log both results
    log_comparison(aws_result, gcp_result)
    
    # Return GCP result (primary) with metadata
    return {
        'prediction': gcp_result,
        'gcp_score': compute_confidence(gcp_result),
        'aws_score': compute_confidence(aws_result),
        'match': np.allclose(gcp_result, aws_result, rtol=1e-2)
    }
```

### 4.4 Regression Testing

```python
def regression_test_suite():
    test_cases = [
        {"name": "test_normal_input", "input": [1.0, 2.0, 3.0], "expected": 2.5},
        {"name": "test_edge_case_zeros", "input": [0, 0, 0], "expected": 0},
        {"name": "test_large_values", "input": [1e6, 2e6], "expected": 1.5e6},
        {"name": "test_negative_values", "input": [-1, -2, -3], "expected": -2}
    ]
    
    for test in test_cases:
        result = call_endpoint(test['input'])
        assert abs(result - test['expected']) < 0.01, f"{test['name']} failed"
```

---

## Phase 5: Deployment

### 5.1 Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Team trained on new platform
- [ ] Rollback plan in place
- [ ] Monitoring alerts configured
- [ ] Cost estimates validated
- [ ] Security policies verified

### 5.2 Deployment Strategies

#### Blue-Green Deployment
```python
# Keep old endpoint (Blue) running
# Deploy new endpoint (Green)
# Route 10% traffic to Green, monitor
# Gradually increase to 50%, then 100%

def blue_green_deployment(model, traffic_split=0.1):
    # Deploy new endpoint (Green)
    green_endpoint = model.deploy(
        machine_type="n1-standard-2",
        traffic_split=traffic_split
    )
    
    # Old endpoint continues (Blue)
    # Traffic distribution: 90% Blue, 10% Green
    return green_endpoint
```

#### Canary Deployment
```python
# Deploy to small subset first
# Monitor for errors
# Gradual rollout

traffic_percentages = [5, 10, 25, 50, 100]

for percentage in traffic_percentages:
    # Update traffic split
    endpoint.update(traffic_split=percentage)
    
    # Monitor for errors
    error_rate = check_error_rate()
    latency = check_latency()
    
    if error_rate > 1% or latency > threshold:
        rollback()
        break
    
    wait(monitoring_period)
```

### 5.3 Deployment Commands

```bash
# Deploy using gcloud CLI
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --model=MODEL_ID \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=3

# Or using Python SDK
from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(endpoint_name="projects/123/locations/us-central1/endpoints/456")
endpoint.deploy(model, machine_type="n1-standard-2", min_replica_count=1, max_replica_count=3)
```

### 5.4 Post-Deployment Validation

```python
# Validate deployment
def validate_deployment(endpoint):
    try:
        # Test prediction
        test_input = [[1.0, 2.0, 3.0]]
        response = endpoint.predict(instances=test_input)
        
        # Check response format
        assert 'predictions' in response
        assert len(response['predictions']) > 0
        
        # Check performance metrics
        metrics = endpoint.get_metrics()
        assert metrics['latency_p95'] < 50  # 50ms
        assert metrics['availability'] > 99.9
        
        print("✓ Deployment validated successfully")
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False
```

### 5.5 Rollback Plan

```python
def rollback_if_needed():
    # Monitor metrics after deployment
    error_rate = check_error_rate()
    latency = check_latency()
    
    if error_rate > 5% or latency > acceptable_threshold:
        print("Rolling back to previous version...")
        
        # Update traffic back to old endpoint
        old_endpoint.update(traffic_split=100)
        new_endpoint.undeploy()
        
        # Alert team
        send_alert("Deployment rolled back due to issues")
        
        return True
    return False
```

---

## Phase 6: Post-Migration Operations

### 6.1 Monitoring & Alerts

```python
from google.cloud import aiplatform

# Setup monitoring
job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name="model-monitoring",
    objective_config=aiplatform.monitoring.ObjectiveConfig(
        deployed_model_id="model-123"
    ),
    alert_config=aiplatform.monitoring.AlertConfig(
        email_alert_config=aiplatform.monitoring.EmailAlertConfig(
            user_emails=["team@company.com"]
        )
    )
)
```

### 6.2 Performance Baseline

Document baseline metrics:
- Average latency: ___ms
- p95 latency: ___ms
- Throughput: ___req/sec
- Error rate: ___%
- Cost per 1M predictions: $___

### 6.3 Decommissioning SageMaker Resources

```bash
# After successful migration and validation period (30+ days):

# 1. Delete endpoints
aws sagemaker delete-endpoint --endpoint-name prod-endpoint

# 2. Delete models
aws sagemaker delete-model --model-name model-v1

# 3. Delete notebooks
aws sagemaker delete-notebook-instance --notebook-instance-name instance-name

# 4. Archive data in S3
aws s3 mv s3://sagemaker-bucket s3://archive-bucket --recursive

# 5. Clean up IAM roles
```

---

## Timeline & Resource Estimates

| Phase | Duration | Resources | Risk Level |
|---|---|---|---|
| Phase 1: Inventory | 1-2 weeks | 1-2 people | Low |
| Phase 2: Mapping | 1-2 weeks | 1-2 people | Low |
| Phase 3: Migration | 2-4 weeks | 2-3 people | Medium |
| Phase 4: Testing | 2-3 weeks | 2-3 people | High |
| Phase 5: Deployment | 1-2 weeks | 3-4 people | High |
| Phase 6: Post-Migration | Ongoing | 1 person | Low |
| **Total** | **8-14 weeks** | **2-4 people** | **Medium** |

---

## Success Criteria

- ✓ All models migrated with <2% accuracy variance
- ✓ Latency within 10% of original
- ✓ 99.9% uptime during and after migration
- ✓ Cost reduction of 10-30%
- ✓ Zero data loss
- ✓ All tests passing
- ✓ Team trained and confident
- ✓ Documentation updated

---

## Common Issues & Troubleshooting

| Issue | Root Cause | Solution |
|---|---|---|
| Data corruption | Transfer interruption | Use gsutil -m (parallel) with checksums |
| Model accuracy drop | Framework differences | Retrain on GCP with same hyperparams |
| Latency increase | Cold starts, smaller instance | Use min_replica_count, warm endpoints |
| High costs | Inefficient scaling | Right-size instances, implement autoscaling |
| Authentication errors | Service account missing | Verify GOOGLE_APPLICATION_CREDENTIALS |

---

## References & Resources

- [GCP Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Vertex AI Python SDK](https://cloud.google.com/python/docs/reference/aiplatform)
- [Migration Guide: ML Models to Vertex AI](https://cloud.google.com/vertex-ai/docs/migration-guide)
- [SageMaker to Vertex AI Mapping](https://cloud.google.com/architecture/migrating-amazon-sagemaker-workloads-to-vertex-ai)

---

**Last Updated:** February 2026
**Version:** 1.0
