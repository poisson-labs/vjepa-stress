# V-JEPA 2.1 stress sweep image.
# NGC PyTorch image bundles CUDA 12.x + PyTorch 2.4+; transformers V-JEPA 2.1
# requires this combination, and the image pulls quickly on RunPod.
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Cache locations all live under /workspace so a persistent RunPod volume
# mounted at /workspace keeps V-JEPA weights, HF assets, and pip's wheel
# cache across container restarts. Without TORCH_HOME the V-JEPA weights
# (~1.5 GB for vit-large, ~6 GB for vit-gigantic) re-download into
# /root/.cache/torch on every container start.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    VJEPA_STRESS_DAVIS_DIR=/workspace/data/davis \
    VJEPA_STRESS_SSV2_DIR=/workspace/data/ssv2

WORKDIR /workspace

# Install project dependencies first so the deps layer caches independently
# of source-only edits. The package is installed editable, so subsequent
# COPYs of scripts/tests/manifests don't trigger a reinstall — only a
# vjepa_stress/ source change re-runs ``pip install -e``, which is fast
# since deps are already resolved and cached.
COPY pyproject.toml README.md /workspace/
COPY vjepa_stress /workspace/vjepa_stress
RUN pip install --upgrade pip \
    && pip install -e ".[dev]"

# Datasets and checkpoints stay outside the image — mounted at runtime via
# a RunPod volume at /workspace/data and /workspace/.cache.
COPY scripts /workspace/scripts
COPY tests /workspace/tests
COPY manifests /workspace/manifests

CMD ["bash"]
