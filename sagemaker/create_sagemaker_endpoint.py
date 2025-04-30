import argparse
import boto3

from pathlib import Path

def create_sagemaker_endpoint(
        region: str,
        instance_type: str,
        instance_count: int,
        role_arn: str,
        image_uri: str,
        endpoint_name: str,
        model_path: str,
        hf_token_path: str,
        sync: bool = False,
        s3_output_path: str = None,
        max_concurrent_invocations_per_instance: int = None,
        max_model_len=None,
        tensor_parallel_size=None,
        data_parallel_size=None,
        gpu_memory_utilization=None,
        swap_space=None,
        disable_custom_all_reduce=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        disable_sliding_window=False,
):
    """
    Create either a synchronous or asynchronous SageMaker endpoint.

    Args:
        sync: If True, create a real-time inference endpoint. Otherwise, async.
        s3_output_path: Required for async endpoints.
        max_concurrent_invocations_per_instance: Required for async endpoints.
    """
    sagemaker = boto3.client('sagemaker', region_name=region)

    # Build environment variables
    env = {
        "API_HOST": "0.0.0.0",
        "API_PORT": "8080",
        "INSTANCE_TYPE": instance_type,
    }

    model_is_on_s3 = model_path.startswith('s3://')
    if model_is_on_s3:
        env["MODEL_ID"] = "/opt/ml/model"
    else:
        hf_token = Path(hf_token_path).read_text().strip()
        env.update({
            "HF_TOKEN": hf_token,
            "HUGGING_FACE_HUB_TOKEN": hf_token,
            "MODEL_ID": model_path,
        })

    # Optional vLLM params
    if max_model_len:
        env['MAX_MODEL_LEN'] = str(max_model_len)
    if tensor_parallel_size:
        env['TENSOR_PARALLEL_SIZE'] = str(tensor_parallel_size)
    if data_parallel_size:
        env['DATA_PARALLEL_SIZE'] = str(data_parallel_size)
    if gpu_memory_utilization:
        env['GPU_MEMORY_UTILIZATION'] = str(gpu_memory_utilization)
    if swap_space:
        env['SWAP_SPACE'] = str(swap_space)
    if disable_custom_all_reduce:
        env['DISABLE_CUSTOM_ALL_REDUCE'] = "1"
    if enable_prefix_caching:
        env['ENABLE_PREFIX_CACHING'] = "1"
    if enable_chunked_prefill:
        env['ENABLE_CHUNKED_PREFILL'] = "1"
    if disable_sliding_window:
        env['DISABLE_SLIDING_WINDOW'] = "1"

    # Create SageMaker model
    model_name = f"{endpoint_name}-model"
    container_def = {
        'Image': image_uri,
        'Environment': env,
    }
    if model_is_on_s3:
        container_def['ModelDataSource'] = {
            'S3DataSource': {
                'S3Uri': model_path,
                'S3DataType': 'S3Prefix',
                'CompressionType': 'None',
            }
        }

    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer=container_def,
        ExecutionRoleArn=role_arn,
    )
    print(f"Created model: {model_name}")

    # Build endpoint config
    config_name = f"{endpoint_name}-config"
    variant = {
        'VariantName': 'default',
        'ModelName': model_name,
        'InstanceType': instance_type,
        'InitialInstanceCount': instance_count,
        'ContainerStartupHealthCheckTimeoutInSeconds': 600,
    }

    endpoint_config_args = {
        'EndpointConfigName': config_name,
        'ProductionVariants': [variant],
    }

    # Add async or sync-specific config
    if not sync:
        if not s3_output_path or not max_concurrent_invocations_per_instance:
            raise ValueError("Async endpoints require --s3_output_path and --max_concurrent_invocations_per_instance.")
        endpoint_config_args['AsyncInferenceConfig'] = {
            'OutputConfig': {'S3OutputPath': s3_output_path},
            'ClientConfig': {'MaxConcurrentInvocationsPerInstance': max_concurrent_invocations_per_instance},
        }

    sagemaker.create_endpoint_config(**endpoint_config_args)
    print(f"Created endpoint config: {config_name}")

    # Create the endpoint
    sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )
    print(f"Creating {'sync' if sync else 'async'} endpoint: {endpoint_name}...")
    print("Check the SageMaker console for status.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint_name', default='vllm-endpoint')
    parser.add_argument('--region', default='us-east-1')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--hf_token_path', default='./huggingface-key')
    parser.add_argument('--instance_type', required=True)
    parser.add_argument('--instance_count', type=int, default=1)
    parser.add_argument('--role_arn', required=True)
    parser.add_argument('--image_uri', required=True)
    parser.add_argument('--sync', action='store_true', help='Create a synchronous real-time endpoint')
    parser.add_argument('--s3_output_path', help='S3 path for async inference output')
    parser.add_argument('--max_concurrent_invocations_per_instance', type=int,
                        help='Max concurrent invocations for async endpoints')
    parser.add_argument('--max_model_len', type=int)
    parser.add_argument('--tensor_parallel_size', type=int, required=True)
    parser.add_argument('--data_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--swap_space', type=int, default=1)
    parser.add_argument('--disable_custom_all_reduce', action='store_true')
    parser.add_argument('--enable_prefix_caching', action='store_true')
    parser.add_argument('--enable_chunked_prefill', action='store_true')
    parser.add_argument('--disable_sliding_window', action='store_true')
    args = parser.parse_args()

    create_sagemaker_endpoint(
        region=args.region,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        role_arn=args.role_arn,
        image_uri=args.image_uri,
        endpoint_name=args.endpoint_name,
        model_path=args.model_path,
        hf_token_path=args.hf_token_path,
        sync=args.sync,
        s3_output_path=args.s3_output_path,
        max_concurrent_invocations_per_instance=args.max_concurrent_invocations_per_instance,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        disable_sliding_window=args.disable_sliding_window,
    )
