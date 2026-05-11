# Stage 1: Build and pack the Conda environment.
# Use a lightweight base — the builder only needs Miniconda + pip; pulling the
# full tritonserver image (11.8 GB) here would double the pull cost for no gain.
FROM python:3.12-slim AS builder

# Install Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/miniconda/bin:$PATH"

# Build context is the pnyx/ parent directory (see build.sh) so both
# pnyx-privacy-filter/ and privacy-filter/ are accessible.
COPY pnyx-privacy-filter/docker/triton/requirements.txt /tmp/requirements.txt
COPY privacy-filter /tmp/privacy-filter

# Create the pnyx_pf environment.
# KEY: install CPU-only torch FIRST (before opf), then install opf with
# --no-deps so pip never pulls the multi-GB CUDA torch distribution.
# All non-torch opf dependencies are listed explicitly in requirements.txt.
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n pnyx_pf python=3.12 -y && \
    /opt/miniconda/envs/pnyx_pf/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch && \
    /opt/miniconda/envs/pnyx_pf/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    /opt/miniconda/envs/pnyx_pf/bin/pip install --no-cache-dir --no-deps /tmp/privacy-filter && \
    /opt/miniconda/bin/conda install -c conda-forge conda-pack -y && \
    conda clean --all -y

# Pack the environment into a tarball.
# --ignore-missing-files: skip conda-managed files overwritten by pip (e.g. setuptools egg-info)
RUN mkdir -p /output && \
    conda-pack -n pnyx_pf --ignore-missing-files -o /output/pnyx_pf.tar.gz

# Stage 2: Final Triton image
FROM tritonserver:latest

# Install gettext-base for envsubst
RUN apt-get update && apt-get install -y gettext-base && \
    rm -rf /var/lib/apt/lists/*

# Create the opt and models directory if it doesn't exist (it should)
RUN mkdir -p /opt
RUN mkdir -p /models

# Copy the packed environment from the builder stage
COPY --from=builder /output/pnyx_pf.tar.gz /opt/pnyx_pf.tar.gz
RUN chmod 644 /opt/pnyx_pf.tar.gz

# Copy model repository configs (templates) to /model_configs
COPY pnyx-privacy-filter/model_repository /model_configs

LABEL description="Triton Inference Server with pnyx_pf python environment"

# The entrypoint and cmd are inherited from the base image, 
# but we can explicitly set them if preferred.
#ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
#CMD ["tritonserver", "--model-repository=/models"]
RUN chmod 777 /model_configs/custom_entrypoint.sh
ENTRYPOINT ["/model_configs/custom_entrypoint.sh"]
