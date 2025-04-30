FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir boto3 sagemaker-inference

# Writable cache so vLLM / HF Hub can create indexes etc.
ENV HF_HOME=/opt/vllm-cache \
    TRANSFORMERS_CACHE=/opt/vllm-cache
RUN mkdir -p /opt/vllm-cache

COPY src/sagemaker_serving.py /app/sagemaker_serving.py
COPY sagemaker/serve          /usr/bin/serve
RUN chmod +x /usr/bin/serve

WORKDIR /app
EXPOSE 8080
ENTRYPOINT ["/usr/bin/serve"]
