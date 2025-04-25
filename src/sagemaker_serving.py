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
    
    # Get arguments from environment or use defaults
    args, _ = parser.parse_known_args()
    
    # Override with SageMaker environment variables
    args.host = os.getenv("API_HOST", "0.0.0.0")
    args.port = int(os.getenv("API_PORT", "8080"))
    args.model = os.getenv("MODEL_ID", args.model)
    
    # Log configuration
    print(f"Starting vLLM server with model: {args.model}")
    print(f"Host: {args.host}, Port: {args.port}")
    
    # Validate arguments
    if args.model is None:
        sys.exit("❌ ERROR: MODEL_ID must be set")
    validate_parsed_serve_args(args)

    # Use the integrated run_server function from vLLM
    asyncio.run(run_server(args))

if __name__ == "__main__":
    start_api_server()
