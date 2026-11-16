# SAGE reproducibility image: the conda env "sage", TeX Live for the figures,
# T5 + TAPEX baked in, and the CoverType CSV unpacked.
#
#   docker build -t sage .                              # CPU (default experiments)
#   docker build -t sage-gpu --build-arg TORCH_VARIANT=cu124 .   # for ARI/run_optional.sh
#   docker run --rm -v "$PWD/docker_out:/sage/ARI/out" sage
# Ubuntu 24.04: its TeX Live 2023 has pgfplots 1.18, which the figures need (20.04's 2019 is too old)
FROM ubuntu:24.04

# cpu (default experiments) or a CUDA build such as cu124 (optional GPU runs)
ARG TORCH_VARIANT=cpu
# 0 skips downloading T5 / TAPEX (~5 GB); the default experiments read the shipped embedding banks
ARG FETCH_MODELS=1
ARG MINIFORGE_VERSION=24.11.3-0

ENV DEBIAN_FRONTEND=noninteractive \
    TRANSFORMER_PATH=/opt/hf \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/conda/bin:$PATH

# TeX Live with pgfplots, libertine, newtx, standalone, pdfpages, booktabs
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        texlive-latex-base texlive-latex-recommended texlive-latex-extra \
        texlive-pictures texlive-plain-generic \
        texlive-fonts-recommended texlive-fonts-extra \
    && rm -rf /var/lib/apt/lists/*

# Miniforge (conda), which the ARI scripts use
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-$(uname -m).sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# The conda env the ARI scripts activate (name "sage"). Torch first, from the chosen index,
# so requirements.txt finds torch==2.6.0 already satisfied.
COPY ARI/env/requirements.txt /tmp/requirements.txt
RUN conda create -y -n sage python=3.10 pip \
    && conda run -n sage pip install --no-cache-dir torch==2.6.0 \
         --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
    && conda run -n sage pip install --no-cache-dir -r /tmp/requirements.txt \
    && conda clean -afy

WORKDIR /sage

# Models in their own layer, so code changes don't re-download them
COPY ARI/scripts/fetch_models.py ARI/scripts/fetch_models.py
RUN if [ "$FETCH_MODELS" = 1 ]; then conda run -n sage python ARI/scripts/fetch_models.py; fi

COPY . .
RUN conda run -n sage python ARI/scripts/unpack_data.py

# "sage" is the default env, also for scripts that don't activate it (smoke_test.sh, python ...)
ENV PATH=/opt/conda/envs/sage/bin:$PATH \
    CONDA_DEFAULT_ENV=sage \
    CONDA_PREFIX=/opt/conda/envs/sage
CMD ["bash", "ARI/run_experiments.sh"]
