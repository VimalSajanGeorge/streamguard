# Custom SageMaker Docker Image for StreamGuard Training
#
# Includes:
# - PyTorch 2.1.0 + CUDA 11.8
# - PyTorch Geometric 2.4.0
# - tree-sitter with pre-built C language support
# - Transformers + tokenizers
#
# Usage:
#   docker build -t streamguard-training:v1 -f Dockerfile .
#   docker tag streamguard-training:v1 YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/streamguard-training:v1
#   docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/streamguard-training:v1

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch Geometric with exact versions
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir \
    torch-geometric==2.4.0 \
    torch-scatter==2.1.2+pt21cu118 \
    torch-sparse==0.6.18+pt21cu118 \
    torch-cluster==1.6.3+pt21cu118 \
    torch-spline-conv==1.2.2+pt21cu118 \
    pyg-lib==0.3.1+pt21cu118 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install Transformers and tokenizers
RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    tokenizers==0.15.0 \
    accelerate==0.24.0

# Install tree-sitter
RUN pip install --no-cache-dir \
    tree-sitter==0.20.4

# Clone tree-sitter-c for AST parsing
WORKDIR /opt/ml/code
RUN mkdir -p vendor && \
    cd vendor && \
    git clone --depth 1 https://github.com/tree-sitter/tree-sitter-c.git

# Pre-build tree-sitter C language library
RUN python3 << 'EOF'
from tree_sitter import Language
from pathlib import Path

build_dir = Path('/opt/ml/code/build')
build_dir.mkdir(parents=True, exist_ok=True)

try:
    Language.build_library(
        str(build_dir / 'my-languages.so'),
        ['/opt/ml/code/vendor/tree-sitter-c']
    )
    print("[+] tree-sitter C language library built successfully")
except Exception as e:
    print(f"[!] Failed to build tree-sitter library: {e}")
EOF

# Install additional dependencies
RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    boto3==1.34.0 \
    sagemaker==2.199.0 \
    tqdm==4.66.1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/opt/ml/model
ENV HF_HOME=/opt/ml/model/huggingface

# Copy training scripts (will be overridden by SageMaker)
COPY training/ /opt/ml/code/training/

# Set working directory
WORKDIR /opt/ml/code

# SageMaker entry point
ENV SAGEMAKER_PROGRAM=train_transformer.py

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python3 -c "import torch; import torch_geometric; import transformers; print('OK')" || exit 1

# Labels
LABEL maintainer="streamguard@example.com"
LABEL version="1.0"
LABEL description="StreamGuard ML Training Environment with PyTorch + PyG + CUDA"
