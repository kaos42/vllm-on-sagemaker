FROM vllm/vllm-openai:latest

# Install any additional requirements
RUN pip install --no-cache-dir \
    boto3 \
    sagemaker-inference

# Copy the new entrypoint scripts
COPY src/sagemaker_serving.py /app/sagemaker_serving.py
COPY sagemaker/serve /usr/bin/serve

# Set the working directory
WORKDIR /app

# Ensure the serve script has executable permissions
RUN chmod +x /usr/bin/serve

# SageMaker uses port 8080
EXPOSE 8080

# Set the serve script as the entrypoint
ENTRYPOINT ["/usr/bin/serve"]
