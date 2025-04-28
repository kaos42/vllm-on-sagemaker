#!/usr/bin/env bash
set -euo pipefail

# ---------- defaults ----------
region="us-east-1"
tag="latest"
image_name="vllm-on-sagemaker"
model_id=""          #  ← now empty by default
hf_token=""

# ---------- parse CLI flags ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)      region="$2";     shift 2;;
    --tag)         tag="$2";        shift 2;;
    --image-name)  image_name="$2"; shift 2;;
    --model-id)    model_id="$2";   shift 2;;
    --hf-token)    hf_token="$2";   shift 2;;
    --) shift; break;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

account=$(aws sts get-caller-identity --query Account --output text)
echo "Building image ${image_name}:${tag}"
[[ -n $model_id ]] && echo "  Model baked in: $model_id" || echo "  Model baked in: none (pull at runtime)"

# ---------- paths ----------
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
docker_file="${script_dir}/Dockerfile"
policy_file="${script_dir}/ecr-policy.json"

registry_suffix=${region/*cn*/.cn}
registry="${account}.dkr.ecr.${region}.amazonaws.com${registry_suffix}"
image_uri="${registry}/${image_name}:${tag}"

# ---------- ensure ECR repo exists ----------
aws ecr describe-repositories --repository-names "$image_name" --region "$region" \
  || aws ecr create-repository --repository-name "$image_name" --region "$region"

aws ecr get-login-password --region "$region" | \
  docker login --username AWS --password-stdin "$registry"

[[ -f $policy_file ]] && aws ecr set-repository-policy \
  --repository-name "$image_name" --policy-text "file://${policy_file}" --region "$region"

# ---------- build & push ----------
docker build --platform linux/amd64 \
  --build-arg MODEL_ID="$model_id" \
  --build-arg HF_TOKEN="$hf_token" \
  -t "$image_name" -f "$docker_file" "$script_dir"

docker tag  "$image_name" "$image_uri"
docker push "$image_uri"

echo "Pushed image: $image_uri"
