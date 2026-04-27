# V-JEPA 2.1 stress sweep image.
# NGC PyTorch image bundles CUDA 12.x + PyTorch 2.4+; transformers V-JEPA 2.1
# requires this combination, and the image pulls quickly on RunPod.
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

WORKDIR /workspace

# Install project dependencies first so layer caches independently of source.
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
RUN pip install --upgrade pip \
    && pip install -e ".[dev]"

# Mount data and checkpoints at runtime; do not bake into the image.
COPY scripts /workspace/scripts
COPY tests /workspace/tests
COPY manifests /workspace/manifests

CMD ["bash"]
