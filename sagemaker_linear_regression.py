"""
Simple AWS SageMaker Linear Regression Module
Demonstrates training and inference with SageMaker's Linear Learner algorithm
"""

import boto3
import numpy as np
import pandas as pd
from sagemaker import Session
from sagemaker.linear_model import LinearLearner
from sagemaker.predictor import csv_serializer, json_deserializer


class SageMakerLinearRegression:
    """Simple wrapper for AWS SageMaker Linear Regression"""
    
    def __init__(self, role_arn, instance_type='ml.m5.large', instance_count=1):
        """
        Initialize SageMaker Linear Regression model
        
        Args:
            role_arn: IAM role ARN for SageMaker
            instance_type: EC2 instance type for training
            instance_count: Number of instances for training
        """
        self.session = Session()
        self.role_arn = role_arn
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.model = None
        self.predictor = None
        
    def prepare_data(self, X_train, y_train, s3_path):
        """
        Prepare and upload training data to S3
        
        Args:
            X_train: Training features (numpy array)
            y_train: Training labels (numpy array)
            s3_path: S3 location to store data (e.g., 's3://bucket-name/path')
        
        Returns:
            S3 URI of uploaded training data
        """
        # SageMaker expects label as first column
        training_data = np.column_stack((y_train, X_train))
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(training_data)
        df.to_csv('/tmp/training_data.csv', header=False, index=False)
        
        # Upload to S3
        s3 = boto3.client('s3')
        bucket_name = s3_path.split('/')[2]
        key = '/'.join(s3_path.split('/')[3:]) + '/training_data.csv'
        
        s3.upload_file('/tmp/training_data.csv', bucket_name, key)
        
        return f'{s3_path}/training_data.csv'
    
    def train(self, s3_training_data, feature_dim, job_name='linear-regression-job'):
        """
        Train the linear regression model
        
        Args:
            s3_training_data: S3 URI of training data
            feature_dim: Number of features in the dataset
            job_name: Name for the training job
        """
        self.model = LinearLearner(
            role=self.role_arn,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            feature_dim=feature_dim,
            predictor_type='regressor',
            epoch=30,
            output_path='s3://{}/output'.format(self.session.default_bucket())
        )
        
        self.model.fit(s3_training_data, job_name=job_name)
        print(f"Training job {job_name} started successfully")
    
    def deploy(self, initial_instance_count=1, instance_type='ml.m5.large'):
        """
        Deploy the trained model as an endpoint
        
        Args:
            initial_instance_count: Number of instances for the endpoint
            instance_type: EC2 instance type for the endpoint
        
        Returns:
            Predictor object for making predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained first using train()")
        
        self.predictor = self.model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type
        )
        
        self.predictor.serializer = csv_serializer
        self.predictor.deserializer = json_deserializer
        
        print("Model deployed successfully")
        return self.predictor
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test: Test features (numpy array or list)
        
        Returns:
            Predictions from the model
        """
        if self.predictor is None:
            raise ValueError("Model must be deployed first using deploy()")
        
        predictions = self.predictor.predict(X_test)
        return predictions
    
    def cleanup(self):
        """Delete the endpoint to stop incurring costs"""
        if self.predictor:
            self.predictor.delete_endpoint()
            print("Endpoint deleted successfully")


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.rand(100, 5)  # 100 samples, 5 features
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] - X_train[:, 2] + np.random.randn(100) * 0.1
    
    print("Sample data generated:")
    print(f"Training samples: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    
    # Note: To run training, you need:
    # 1. AWS credentials configured
    # 2. IAM role ARN with SageMaker permissions
    # 3. S3 bucket for storing data and model artifacts
    
    # Uncomment and update the following to train:
    # role_arn = "arn:aws:iam::ACCOUNT_ID:role/SageMakerRole"
    # s3_bucket = "s3://your-bucket-name"
    # 
    # model = SageMakerLinearRegression(role_arn)
    # s3_data_path = model.prepare_data(X_train, y_train, s3_bucket)
    # model.train(s3_data_path, feature_dim=5)
    # model.deploy()
    # 
    # predictions = model.predict(X_train[:5])
    # print(f"Predictions: {predictions}")
    # 
    # model.cleanup()
