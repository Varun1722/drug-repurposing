# TxGNN needs Python 3.8 and DGL 0.5.2, which has no Apple-Silicon build, so run it as linux/amd64.
FROM --platform=linux/amd64 python:3.8-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# CPU PyTorch 1.7.1 is the newest CPU build that DGL 0.5.2's 0.5 line was built against.
RUN pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
 && pip install dgl==0.5.2 numpy==1.23.5 pandas==1.5.3 scikit-learn==1.1.3 \
        tqdm matplotlib goatools pyarrow==12.0.1

# TxGNN pinned to a commit (main as of 2026-10-07; last commit 2024-05-06). --no-deps keeps
# the versions above.
ARG TXGNN_COMMIT=0e0203c3939f7d12da27aaf10e55daaff5424023
RUN pip install --no-deps https://github.com/mims-harvard/TxGNN/archive/${TXGNN_COMMIT}.zip

WORKDIR /work
