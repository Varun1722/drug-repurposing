# CPU-only gnina runtime for Modal. Uses gnina's official prebuilt
# "somewhat static" release binary (openbabel/boost/libtorch are statically
# linked in) instead of compiling from source or vendoring the ~60GB
# official gnina/gnina CUDA build image.
#
# The binary is still dynamically linked against 7 CUDA/cuDNN runtime
# libraries even though it never touches a GPU when run with --no_gpu:
#   libcudnn.so.9  libcudart.so.12  libcusparse.so.12  libcufft.so.11
#   libcublas.so.12  libcublasLt.so.12  libcusolver.so.11
# (confirmed via `ldd` against the release binary). Those are exactly what
# PyTorch's own cu12 pip wheels ship, so we grab them from PyPI instead of
# pulling any CUDA/Docker image -- no GPU, no driver, no nvidia base image
# anywhere in this Dockerfile.
#
# Validated: gnina --no_gpu docks gnina's own 184L reference ligand/receptor
# pair cleanly in a container with zero GPU/driver present.

FROM --platform=linux/amd64 python:3.11-slim AS cuda-libs-builder

RUN pip install --no-cache-dir --default-timeout=180 --retries 5 \
      nvidia-cudnn-cu12 \
      nvidia-cublas-cu12 \
      nvidia-cusparse-cu12 \
      nvidia-cufft-cu12 \
      nvidia-cusolver-cu12 \
      nvidia-cuda-runtime-cu12 \
    && mkdir -p /cuda-libs \
    && find /usr/local/lib/python3.11/site-packages/nvidia -name '*.so*' -exec cp -L {} /cuda-libs/ \;

# fpocket isn't packaged for Debian/apt (confirmed: `apt-get install fpocket`
# 404s on python:3.11-slim), and its own source (github.com/Discngine/fpocket,
# last touched years ago) fails to compile under GCC 14's default strictness
# -- fparams.c passes `&some_array` where a plain pointer is expected, which
# older GCC only warned about but 14+ treats as a hard error. Rather than
# patching that abandoned C codebase, use the prebuilt conda-forge package
# (anaconda.org/conda-forge/fpocket, linux-64, ~1.7MB) via micromamba -- a
# single static binary that needs no Python/conda install of its own, used
# here only to resolve and fetch that one package plus its runtime deps.
FROM --platform=linux/amd64 python:3.11-slim AS fpocket-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C /usr/local bin/micromamba \
    && micromamba create -y --override-channels -c conda-forge -p /opt/fpocket-env fpocket

FROM --platform=linux/amd64 python:3.11-slim AS runtime

COPY --from=cuda-libs-builder /cuda-libs/ /usr/local/lib/cuda-libs/
RUN echo /usr/local/lib/cuda-libs > /etc/ld.so.conf.d/cuda-libs.conf && ldconfig

ADD https://github.com/gnina/gnina/releases/latest/download/gnina.cuda12.8.static /usr/local/bin/gnina
RUN chmod +x /usr/local/bin/gnina

# fpocket: Tier 2 of the binding-site box fallback (predicted pocket, used
# only when the receptor has no co-crystallized ligand). The conda-forge
# package pulls in its own runtime deps (netcdf, boost, etc.) into the same
# env prefix rather than relying on system apt packages, so the whole env's
# lib/ dir is registered with the dynamic linker alongside the binary --
# same pattern already used above for the CUDA runtime libs.
# This remains a soft dependency: gnina_worker.py's _pocket_box() wraps
# every call in try/except and falls through to blind docking on any
# failure, so even a broken/missing fpocket binary degrades Tier 2 rather
# than breaking a docking request.
COPY --from=fpocket-builder /opt/fpocket-env/bin/fpocket /usr/local/bin/fpocket
COPY --from=fpocket-builder /opt/fpocket-env/lib/ /usr/local/lib/fpocket-libs/
RUN echo /usr/local/lib/fpocket-libs > /etc/ld.so.conf.d/fpocket-libs.conf && ldconfig
