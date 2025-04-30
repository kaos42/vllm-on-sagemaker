#!/usr/bin/env python3
"""
Script to download a model from Hugging Face Hub to a local directory.
Uses environment variables:
- MODEL_ID: Required. The Hugging Face model ID (e.g., "mistralai/Mistral-7B-v0.1")
- HF_TOKEN: Optional. Hugging Face access token for private models
"""

import os
import sys
from huggingface_hub import snapshot_download


def download_model():
    model_id = os.environ.get("MODEL_ID")
    if not model_id:
        print("ERROR: MODEL_ID environment variable not set")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")

    print(f">> Downloading model {model_id} to /opt/models")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir="/opt/models",
            local_dir_use_symlinks=False,
            resume_download=True,
            token=hf_token if hf_token else None,
        )
        print(f">> Successfully downloaded model {model_id}")
    except Exception as e:
        print(f"ERROR downloading model: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
