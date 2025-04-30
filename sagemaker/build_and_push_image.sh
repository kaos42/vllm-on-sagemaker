#!/usr/bin/env bash
set -euo pipefail

# ---------- defaults ----------
region="us-east-1"
tag="latest"
image_name="vllm-on-sagemaker"

# ---------- parse CLI flags ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)      region="$2";     shift 2;;
    --tag)         tag="$2";        shift 2;;
    --image-name)  image_name="$2"; shift 2;;
    --) shift; break;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

account=$(aws sts get-caller-identity --query Account --output text)
echo "Building image ${image_name}:${tag}"

# ---------- paths ----------
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
docker_file="${script_dir}/Dockerfile"
policy_file="${script_dir}/ecr-policy.json"

# ---------- registry URL ----------
registry="${account}.dkr.ecr.${region}.amazonaws.com"
if [[ $region == *cn-* ]]; then
  registry="${registry}.cn"
fi
image_uri="${registry}/${image_name}:${tag}"


# ---------- ensure ECR repo exists ----------
aws ecr describe-repositories --repository-names "$image_name" --region "$region" \
  || aws ecr create-repository --repository-name "$image_name" --region "$region"

aws ecr get-login-password --region "$region" | \
  docker login --username AWS --password-stdin "$registry"

[[ -f $policy_file ]] && aws ecr set-repository-policy \
  --repository-name "$image_name" --policy-text "file://${policy_file}" --region "$region"

# ---------- build & push ----------
# Get parent directory (project root) for build context
build_context="$(dirname "$script_dir")"

docker build --platform linux/amd64 \
  -t "$image_name" -f "$docker_file" "$build_context"

docker tag  "$image_name" "$image_uri"
docker push "$image_uri"

echo "Pushed image: $image_uri"
