# Lab PyTorch image (torch 2.1 / CUDA 12.1) has no kernels for RTX 50xx (sm_120).
# This layer swaps in a CUDA 12.8 build of PyTorch and adds the notebook's dependencies.
# Build:  docker build -t pytorch-task1 task1_llm/member_anushka/src
FROM pytorch:latest
RUN pip install --no-cache-dir --upgrade torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir datasets tqdm matplotlib nbconvert nbclient ipykernel
