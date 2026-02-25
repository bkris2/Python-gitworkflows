"""
GCP Vertex AI template equivalent for Engagement Propensity model.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from google.cloud import aiplatform


@dataclass
class EngagementPropensityVertexConfig:
    project_id: str
    region: str
    staging_bucket: str
    service_account: Optional[str] = None
    machine_type: str = "n1-standard-4"


class EngagementPropensityVertexTemplate:
    def __init__(self, config: EngagementPropensityVertexConfig):
        self.config = config
        self.model = None
        self.endpoint = None
        aiplatform.init(
            project=self.config.project_id,
            location=self.config.region,
            staging_bucket=self.config.staging_bucket,
        )

    def default_args(self) -> Dict[str, str]:
        return {
            "model_type": "engagement_propensity",
            "target_column": "engaged",
            "algorithm": "xgboost",
            "max_depth": "6",
            "learning_rate": "0.05",
            "n_estimators": "300",
        }

    def run_custom_training(
        self,
        display_name: str,
        script_path: str,
        container_uri: str,
        train_args: Optional[Dict[str, str]] = None,
        model_serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    ):
        job = aiplatform.CustomContainerTrainingJob(
            display_name=display_name,
            container_uri=container_uri,
            model_serving_container_image_uri=model_serving_container_image_uri,
        )

        args = train_args or self.default_args()
        self.model = job.run(
            model_display_name=f"{display_name}-model",
            args=[f"--{k}={v}" for k, v in args.items()] + [f"--script_path={script_path}"],
            machine_type=self.config.machine_type,
            service_account=self.config.service_account,
            replica_count=1,
            sync=False,
        )
        return self.model

    def deploy(self, endpoint_display_name: str, machine_type: str = "n1-standard-2"):
        if self.model is None:
            raise ValueError("Run training before deploy().")
        self.endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
        self.model.deploy(
            endpoint=self.endpoint,
            machine_type=machine_type,
            min_replica_count=1,
            max_replica_count=2,
            traffic_split={"0": 100},
        )
        return self.endpoint
