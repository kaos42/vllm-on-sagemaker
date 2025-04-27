# SageMaker Endpoint for vLLM

_forked from github.com/JianyuZhan/vllm-on-sagemaker_

You can use the [LMI](https://docs.djl.ai/docs/serving/serving/docs/lmi/index.html) to easily run vLLM on Amazon SageMaker. However, the version of vLLM supported by LMI lags several versions behind the latest community version. If you want to run the latest version, try this repo!

## Prerequisites

Make sure you have the following tools installed:
- AWS CLI (and run `aws configure`)
- Docker
- Python 3

## Usage

### 1. Set Environment Variables

Start by setting up some environment variables. Adjust them as needed:

```sh
export REGION='us-east-1' # change as needed
export IMG_NAME='vllm-on-sagemaker' # change as needed
export IMG_TAG='latest' # change as needed
export SAGEMAKER_ENDPOINT_NAME='vllm-on-sagemaker' # change as needed
```

### 2. Build and Push Docker Image

Build the Docker image that will be used to run the SageMaker Endpoint serving container. After building, the image will be pushed to AWS ECR. The container implements `/ping` and `/invocations` APIs, as required by SageMaker Endpoints.

```sh
sagemaker/build_and_push_image.sh --region "$REGION" --image-name "$IMG_NAME" --tag "$IMG_TAG"
```

### 3. Get the Image URI

After the image is built and pushed, retrieve the image URI:

```sh
export IMG_URI=$(sagemaker/get_ecr_image_uri.sh --region "$REGION" --img-name "$IMG_NAME" --tag "$IMG_TAG")
echo $IMG_URI
```

### 4. Create a SageMaker Execution Role

Create a SageMaker execution role to allow the endpoint to run properly:

```sh
export SM_ROLE=$(sagemaker/create_sagemaker_execute_role.sh)
echo $SM_ROLE
```

### 5. Create the SageMaker Endpoint

Now, create the SageMaker Endpoint. Choose the appropriate Hugging Face model ID and instance type:

```sh
# Change various arguments as needed
python sagemaker/create_sagemaker_endpoint.py \
  --region ${REGION} \
  --model_id Qwen/Qwen2.5-32B-Instruct \
  --instance_type ml.g6.48xlarge \
  --instance_count 10 \
  --role_arn ${SM_ROLE} \
  --image_uri ${IMG_URI} \
  --endpoint_name ${SAGEMAKER_ENDPOINT_NAME} \
  --s3_output_path "my-bucket/sagemaker-output" \
  --max_concurrent_invocations_per_instance 50 \
  --max_model_len 5120 \
  --tensor_parallel_size 8 \
  --gpu_memory_utilization 0.8 \
  --swap_space 8 \
  --disable_custom_all_reduce \
  --enable_prefix_caching \
  --disable_sliding_window
```

Edit 04/26/2025: Updated `create_sagemaker_endpoint` to create an async Sagemaker endpoint. If you want to create a sync endpoint, remove `AsyncInferenceConfig` from the endpoint config creation, and `--max_concurrent_invocations_per_instance` and `--s3_output_path` above.

### 6. Check the Endpoint

Go to the AWS console -> SageMaker -> Inference -> Endpoints. You should see the endpoint being created. Wait until the creation process is complete.

### 7. Send Requests to the Endpoint

Once the endpoint is created and in 'InService' status, you can start sending requests to it. Note that for async endpoint config you'll need to use the InvokeEndpointAsync API.
