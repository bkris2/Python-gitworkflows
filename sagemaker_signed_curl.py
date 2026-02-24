#!/usr/bin/env python3
"""
sagemaker_signed_curl.py

Generate a SigV4-signed curl command to invoke a SageMaker endpoint.

Usage:
  python sagemaker_signed_curl.py --endpoint prod-endpoint --region us-east-1 --payload payload.json

Requires: boto3 (botocore)
"""
import argparse
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


def build_signed_curl(endpoint_name, region, payload_file, content_type="application/json"):
    with open(payload_file, 'rb') as f:
        payload = f.read()

    url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"

    session = boto3.Session()
    creds = session.get_credentials().get_frozen_credentials()

    aws_request = AWSRequest(method='POST', url=url, data=payload, headers={
        'Content-Type': content_type
    })

    SigV4Auth(creds, 'sagemaker', region).add_auth(aws_request)

    headers = dict(aws_request.headers)

    # Build curl parts
    curl_parts = ["curl -s -X POST"]
    for k, v in headers.items():
        # Escape single quotes in header values
        v_escaped = v.replace("'", "'\\''")
        curl_parts.append(f"-H '{k}: {v_escaped}'")

    curl_parts.append(f"--data-binary @{payload_file}")
    curl_parts.append(f"'{url}'")

    return " \\\n        ".join(curl_parts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a SigV4-signed curl command for SageMaker invoke')
    parser.add_argument('--endpoint', required=True, help='SageMaker endpoint name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--payload', required=True, help='Path to payload file (JSON or raw)')
    parser.add_argument('--content-type', default='application/json', help='Content-Type header')

    args = parser.parse_args()

    try:
        curl_cmd = build_signed_curl(args.endpoint, args.region, args.payload, args.content_type)
        print('\n# Run this command in your shell to invoke the endpoint:')
        print(curl_cmd)
    except Exception as e:
        print(f'Error: {e}')
        raise
