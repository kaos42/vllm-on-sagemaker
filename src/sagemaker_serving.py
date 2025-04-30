#!/usr/bin/env python3
import asyncio
import os
import sys

from vllm.entrypoints.openai.api_server import run_server
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

def start_api_server():
    # Parse command-line arguments with defaults
    parser = make_arg_parser(FlexibleArgumentParser())
    
    # First parse known args
    args, _ = parser.parse_known_args()
    
    # Override with SageMaker environment variables (basic settings)
    args.host = os.getenv("API_HOST", "0.0.0.0")
    args.port = int(os.getenv("API_PORT", "8080"))
    # If MODEL_ID is provided, it means that we will pull the model from HF. Otherwise, it means Sagemaker should
    # get the model downloaded to its default model dir
    args.model = os.getenv("MODEL_ID") or os.getenv("SM_MODEL_DIR")

    # Add the vLLM advanced options from environment variables
    if os.getenv("MAX_MODEL_LEN"):
        args.max_model_len = int(os.getenv("MAX_MODEL_LEN"))
    
    if os.getenv("TENSOR_PARALLEL_SIZE"):
        args.tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE"))
    
    if os.getenv("GPU_MEMORY_UTILIZATION"):
        args.gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION"))
    
    if os.getenv("SWAP_SPACE"):
        args.swap_space = int(os.getenv("SWAP_SPACE"))
    
    # Boolean flags - if the env var exists, enable/disable the feature
    if os.getenv("DISABLE_CUSTOM_ALL_REDUCE", "").lower() in ("1", "true", "yes"):
        args.disable_custom_all_reduce = True
        
    if os.getenv("ENABLE_PREFIX_CACHING", "").lower() in ("1", "true", "yes"):
        args.enable_prefix_caching = True
        
    if os.getenv("DISABLE_SLIDING_WINDOW", "").lower() in ("1", "true", "yes"):
        args.disable_sliding_window = True

    if os.getenv("ENABLE_CHUNKED_PREFILL", "").lower() in ("1", "true", "yes"):
        args.enable_chunked_prefill = True
    
    # Log configuration
    print(f"Starting vLLM server with model/path: {args.model}")
    print(f"Host: {args.host}, Port: {args.port}")
    print(f"Advanced options:")
    print(f"  Max Model Length: {getattr(args, 'max_model_len', 'default')}")
    print(f"  Tensor Parallel Size: {getattr(args, 'tensor_parallel_size', 'default')}")
    print(f"  GPU Memory Utilization: {getattr(args, 'gpu_memory_utilization', 'default')}")
    print(f"  Swap Space: {getattr(args, 'swap_space', 'default')}")
    print(f"  Custom All Reduce: {'disabled' if getattr(args, 'disable_custom_all_reduce', False) else 'enabled'}")
    print(f"  Prefix Caching: {'enabled' if getattr(args, 'enable_prefix_caching', False) else 'disabled'}")
    print(f"  Sliding Window: {'disabled' if getattr(args, 'disable_sliding_window', False) else 'enabled'}")
    
    # Validate arguments
    if args.model is None:
        sys.exit("❌ ERROR: neither MODEL_ID nor SM_MODEL_DIR is set")
    validate_parsed_serve_args(args)

    # Use the integrated run_server function from vLLM
    asyncio.run(run_server(args))

if __name__ == "__main__":
    start_api_server()
