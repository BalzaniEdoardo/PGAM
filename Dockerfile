FROM python:3.11-slim

LABEL maintainer="Edoardo Balzani"

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential : gcc/g++ for Cython extensions (OpenMP via -fopenmp)
# libgomp1        : OpenMP runtime linked by the Cython .so files
# r-base          : R interpreter required by rpy2
# pkg-config      : needed by some pip source builds (e.g. h5py)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        r-base \
        r-base-dev \
        pkg-config \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# ── R packages ────────────────────────────────────────────────────────────────
# survey       : required by PGAM p-value computation
# CompQuadForm : distribution of quadratic forms; PGAM falls back to
#                saddlepoint approximation when absent, but better to have it
RUN R -e "install.packages(c('survey', 'CompQuadForm'), \
          dependencies=TRUE, repos='http://cran.rstudio.com/')"

# ── Python dependencies ───────────────────────────────────────────────────────
# All packages installed via pip — works natively on both amd64 and arm64
# without the SSL/QEMU issues that affect conda in emulated builds.
# rpy2 is built from source (--no-binary rpy2-rinterface) so it links against
# the system R installed above.
RUN pip install --no-cache-dir \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        h5py \
        pyyaml \
        numba

RUN pip install --no-cache-dir \
        jupyter \
        jupyterlab \
        jupytext

RUN pip install --no-cache-dir \
        "rpy2" \
        --no-binary rpy2-rinterface

RUN pip install --no-cache-dir \
        opt-einsum \
        statsmodels \
        dill \
        csr \
        Cython

# ── Install PGAM package ──────────────────────────────────────────────────────
# The build context is the repo root (PGAM/).  We copy everything in, run
# pip install (which invokes setup.py and compiles the Cython extensions),
# then remove the build tree.
COPY . /build/PGAM/
RUN pip install --no-cache-dir /build/PGAM/ && \
    rm -rf /build

# ── Workspace layout ─────────────────────────────────────────────────────────
RUN mkdir -p /notebooks /output /scripts /input /config

# Jupyter server config (no token, allow root, notebook dir = /notebooks)
COPY conf/.jupyter /root/.jupyter

# Tutorial notebooks bundled in the image
COPY PGAM_Tutorial.ipynb        /notebooks/
COPY island_col_mask_demo.ipynb /notebooks/

COPY run_jupyter.sh /
RUN chmod +x /run_jupyter.sh

# Matplotlib non-interactive backend (no display in container)
ENV MPLBACKEND=Agg

EXPOSE 8888 6006

VOLUME ["/notebooks", "/output", "/scripts"]

CMD ["/run_jupyter.sh"]
