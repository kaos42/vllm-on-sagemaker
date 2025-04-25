#!/bin/bash
# cleanup_ecr.sh

# Set default values
REPOSITORY_NAME="vllm-on-sagemaker"
REGION="us-east-1"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --repository-name)
      REPOSITORY_NAME="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Deleting all images in repository: $REPOSITORY_NAME"

# List and delete all images in the repository
IMAGE_IDS=$(aws ecr list-images --repository-name $REPOSITORY_NAME --region $REGION --query 'imageIds[*]' --output json)

if [ ! -z "$IMAGE_IDS" ] && [ "$IMAGE_IDS" != "[]" ]; then
  echo "Deleting images from ECR repository..."
  aws ecr batch-delete-image --repository-name $REPOSITORY_NAME --region $REGION --image-ids "$IMAGE_IDS" || echo "Failed to delete images"
else
  echo "No images found in repository"
fi

# Delete the repository
echo "Deleting ECR repository: $REPOSITORY_NAME"
aws ecr delete-repository --repository-name $REPOSITORY_NAME --region $REGION --force || echo "Failed to delete repository"

echo "ECR cleanup complete"
