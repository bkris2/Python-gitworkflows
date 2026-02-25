"""
AWS SageMaker template for Engagement Propensity model.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from sagemaker.sklearn.estimator import SKLearn


@dataclass
class EngagementPropensitySageMakerConfig:
    role_arn: str
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1
    framework_version: str = "1.2-1"
    py_version: str = "py3"
    output_path: Optional[str] = None


class EngagementPropensitySageMakerTemplate:
    def __init__(self, config: EngagementPropensitySageMakerConfig):
        self.config = config
        self.estimator: Optional[SKLearn] = None

    def default_hyperparameters(self) -> Dict[str, str]:
        return {
            "model_type": "engagement_propensity",
            "target_column": "engaged",
            "algorithm": "xgboost",
            "max_depth": "6",
            "learning_rate": "0.05",
            "n_estimators": "300",
        }

    def create_estimator(self, entry_point: str, source_dir: str, hyperparameters: Optional[Dict[str, str]] = None) -> SKLearn:
        self.estimator = SKLearn(
            entry_point=entry_point,
            source_dir=source_dir,
            role=self.config.role_arn,
            instance_type=self.config.instance_type,
            instance_count=self.config.instance_count,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            output_path=self.config.output_path,
            hyperparameters=hyperparameters or self.default_hyperparameters(),
        )
        return self.estimator

    def train(self, train_s3_uri: str, validation_s3_uri: str, job_name: str) -> None:
        if self.estimator is None:
            raise ValueError("Call create_estimator() before train().")
        self.estimator.fit(
            inputs={"train": train_s3_uri, "validation": validation_s3_uri},
            job_name=job_name,
            wait=False,
        )

    def deploy(self, endpoint_name: str, instance_type: str = "ml.m5.large", initial_instance_count: int = 1):
        if self.estimator is None:
            raise ValueError("Call create_estimator() and train() before deploy().")
        return self.estimator.deploy(
            endpoint_name=endpoint_name,
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
        )
