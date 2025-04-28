##############################
#  Dockerfile
##############################
FROM vllm/vllm-openai:latest

# ---------- build-time arguments ----------
# Leave MODEL_ID blank to skip pre-download and keep legacy behaviour.
ARG MODEL_ID=""
ARG HF_TOKEN=""

ENV MODEL_ID=${MODEL_ID}

# ---------- extra Python deps ----------
RUN pip install --no-cache-dir \
        boto3 \
        sagemaker-inference

# ---------- optionally pre-download the model ----------
RUN if [ -n "$MODEL_ID" ]; then \
        echo ">> Baking model $MODEL_ID into the image"; \
        python - <<'PY' ;\
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    local_dir="/opt/models",
    local_dir_use_symlinks=False,
    resume_download=True,
    token=os.environ.get("HF_TOKEN") or None,
)
PY
        # expose cache only when it exists
        echo 'ENV HF_HOME=/opt/models'       >> /etc/profile.d/hf_cache.sh && \
        echo 'ENV TRANSFORMERS_CACHE=/opt/models' >> /etc/profile.d/hf_cache.sh ; \
    else \
        echo ">> No MODEL_ID supplied – will pull model at runtime"; \
    fi

# ---------- SageMaker entry point ----------
COPY src/sagemaker_serving.py  /app/sagemaker_serving.py
COPY sagemaker/serve           /usr/bin/serve
RUN chmod +x /usr/bin/serve

WORKDIR /app
EXPOSE 8080
ENTRYPOINT ["/usr/bin/serve"]
