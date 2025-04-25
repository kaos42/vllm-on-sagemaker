import boto3
import argparse

def create_sagemaker_endpoint(region, instance_type, role_arn, image_uri, endpoint_name, model_id):
    sagemaker = boto3.client('sagemaker', region_name=region)

    # Create the model
    create_model_response = sagemaker.create_model(
        ModelName=endpoint_name + '-model',
        PrimaryContainer={
            'Image': image_uri,
            'Environment': {
                'API_HOST': '0.0.0.0',
                'API_PORT': '8080',
                'MODEL_ID': model_id,
                'INSTANCE_TYPE': instance_type,
            },
        },
        ExecutionRoleArn=role_arn,
    )
    print(f"Created model: {endpoint_name}-model")

    # Create the endpoint configuration
    create_endpoint_config_response = sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_name + '-config',
        ProductionVariants=[
            {
                'VariantName': 'default',
                'ModelName': endpoint_name + '-model',
                'InstanceType': instance_type,
                'InitialInstanceCount': 1,
                'ContainerStartupHealthCheckTimeoutInSeconds': 600,  # 10 minutes for model loading
            },
        ],
    )
    print(f"Created endpoint config: {endpoint_name}-config")

    # Create the endpoint
    create_endpoint_response = sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_name + '-config',
    )
    print(f"Creating endpoint: {endpoint_name} - this may take some time...")
    print(f"Check the SageMaker console for endpoint status.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--model_id', required=True, help='Hugging Face model ID')
    parser.add_argument('--instance_type', required=True, help='SageMaker instance type')
    parser.add_argument('--role_arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--image_uri', required=True, help='ECR image URI')
    parser.add_argument('--endpoint_name', default='vllm-endpoint', help='SageMaker endpoint name')

    args = parser.parse_args()

    create_sagemaker_endpoint(
        region=args.region,
        instance_type=args.instance_type,
        role_arn=args.role_arn,
        image_uri=args.image_uri,
        endpoint_name=args.endpoint_name,
        model_id=args.model_id,
    )
