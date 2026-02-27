FROM continuumio/miniconda3:latest

LABEL maintainer="Edoardo Balzani"

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential : gcc/g++ for Cython extensions (OpenMP via -fopenmp)
# libgomp1        : OpenMP runtime linked by the Cython .so files
# r-base          : R interpreter required by rpy2
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        r-base \
    && rm -rf /var/lib/apt/lists/*

# ── R packages ────────────────────────────────────────────────────────────────
# survey       : required by PGAM p-value computation
# CompQuadForm : distribution of quadratic forms; PGAM falls back to
#                saddlepoint approximation when absent, but better to have it
RUN R -e "install.packages(c('survey', 'CompQuadForm'), \
          dependencies=TRUE, repos='http://cran.rstudio.com/')"

# ── Python 3.11 + conda-managed binary deps ──────────────────────────────────
RUN /opt/conda/bin/conda install -y python=3.11 && \
    /opt/conda/bin/conda clean -afy

RUN /opt/conda/bin/conda install -y \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        jupyter \
        jupyterlab \
        numba \
        h5py \
        pyyaml \
    && /opt/conda/bin/conda clean -afy

# ── pip-only / source-built deps ─────────────────────────────────────────────
# rpy2 must be compiled against the system R; --no-binary rpy2-rinterface
# forces a source build that picks up /usr/bin/R automatically.
RUN pip install --no-cache-dir \
        "rpy2" \
        --no-binary rpy2-rinterface

RUN pip install --no-cache-dir \
        opt-einsum \
        statsmodels \
        dill \
        jupytext \
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
