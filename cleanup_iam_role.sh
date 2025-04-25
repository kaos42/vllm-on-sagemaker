#!/bin/bash
# cleanup_iam_role.sh

ROLE_NAME="SageMakerExecutionRoleTest"

# Detach policies
echo "Detaching policies from role: $ROLE_NAME"
aws iam detach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam detach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
aws iam detach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

# Delete role
echo "Deleting role: $ROLE_NAME"
aws iam delete-role --role-name $ROLE_NAME

echo "IAM role cleanup complete"
