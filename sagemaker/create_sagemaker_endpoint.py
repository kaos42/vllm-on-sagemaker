import argparse
import boto3


def create_sagemaker_endpoint(
        region: str,
        instance_type: str,
        instance_count: int,
        role_arn: str,
        image_uri: str,
        endpoint_name: str,
        model_id: str,
        s3_output_path: str,
        max_concurrent_invocations_per_instance: int,
        max_model_len=None,
        tensor_parallel_size=None,
        gpu_memory_utilization=None,
        swap_space=None,
        disable_custom_all_reduce=False,
        enable_prefix_caching=False,
        disable_sliding_window=False,
):
    """

    :param region:
    :param instance_type:
    :param instance_count:
    :param role_arn:
    :param image_uri:
    :param endpoint_name:
    :param model_id:
    :param s3_output_path:
    :param max_concurrent_invocations_per_instance:
    :param max_model_len:
    :param tensor_parallel_size:
    :param gpu_memory_utilization:
    :param swap_space:
    :param disable_custom_all_reduce:
    :param enable_prefix_caching:
    :param disable_sliding_window:
    :return:

    Example:
    python sagemaker/create_sagemaker_endpoint.py \
      --region ${REGION} \
      --model_id Qwen/Qwen2.5-32B-Instruct \
      --instance_type ml.g6.48xlarge \
      --instance_count 10 \
      --role_arn ${SM_ROLE} \
      --image_uri ${IMG_URI} \
      --endpoint_name ${SAGEMAKER_ENDPOINT} \
      --max_concurrent_invocations_per_instance 50 \
      --max_model_len 5120 \
      --tensor_parallel_size 8 \
      --gpu_memory_utilization 0.8 \
      --swap_space 8 \
      --disable_custom_all_reduce \
      --enable_prefix_caching \
      --disable_sliding_window
    """
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    # Build environment variables dictionary
    env = {
        'API_HOST': '0.0.0.0',
        'API_PORT': '8080',
        'MODEL_ID': model_id,
        'INSTANCE_TYPE': instance_type,
    }
    
    # Add optional vLLM parameters if provided
    if max_model_len:
        env['MAX_MODEL_LEN'] = str(max_model_len)
    
    if tensor_parallel_size:
        env['TENSOR_PARALLEL_SIZE'] = str(tensor_parallel_size)
    
    if gpu_memory_utilization:
        env['GPU_MEMORY_UTILIZATION'] = str(gpu_memory_utilization)
    
    if swap_space:
        env['SWAP_SPACE'] = str(swap_space)
    
    if disable_custom_all_reduce:
        env['DISABLE_CUSTOM_ALL_REDUCE'] = "1"
    
    if enable_prefix_caching:
        env['ENABLE_PREFIX_CACHING'] = "1"
    
    if disable_sliding_window:
        env['DISABLE_SLIDING_WINDOW'] = "1"

    # Create the model
    create_model_response = sagemaker.create_model(
        ModelName=endpoint_name + '-model',
        PrimaryContainer={
            'Image': image_uri,
            'Environment': env,
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
                'InitialInstanceCount': instance_count,
                'ContainerStartupHealthCheckTimeoutInSeconds': 600,  # 10 minutes for model loading
            },
        ],
        AsyncInferenceConfig={
            'OutputConfig': {'S3OutputPath': s3_output_path},
            'ClientConfig': {'MaxConcurrentInvocationsPerInstance': max_concurrent_invocations_per_instance},
        },
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
    parser.add_argument('--instance_count', type=int, default=1, help='SageMaker instance count')
    parser.add_argument('--role_arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--image_uri', required=True, help='ECR image URI')
    parser.add_argument('--endpoint_name', default='vllm-endpoint', help='SageMaker endpoint name')

    # async options
    parser.add_argument('--s3_output_path', default='s3://veddeshp-302263071703/sagemaker-async-output/', help='S3 path to where async results will be output')
    parser.add_argument('--max_concurrent_invocations_per_instance', type=int, default=10, help='Number of requests Sagemaker will send to each container before throttling new ones')
    
    # Add vLLM specific options
    parser.add_argument('--max_model_len', type=int, help='Maximum sequence length')
    parser.add_argument('--tensor_parallel_size', type=int, help='Number of GPUs for tensor parallelism')
    parser.add_argument('--gpu_memory_utilization', type=float, help='Fraction of GPU memory to use (0.0-1.0)')
    parser.add_argument('--swap_space', type=int, help='CPU swap space in GiB')
    parser.add_argument('--disable_custom_all_reduce', action='store_true', help='Disable custom all-reduce implementation')
    parser.add_argument('--enable_prefix_caching', action='store_true', help='Enable prefix caching')
    parser.add_argument('--disable_sliding_window', action='store_true', help='Disable sliding window attention')

    args = parser.parse_args()

    create_sagemaker_endpoint(
        region=args.region,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        role_arn=args.role_arn,
        image_uri=args.image_uri,
        endpoint_name=args.endpoint_name,
        model_id=args.model_id,
        s3_output_path=args.s3_output_path,
        max_concurrent_invocations_per_instance=args.max_concurrent_invocations_per_instance,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_sliding_window=args.disable_sliding_window,
    )
