# PyTorch 2.8 / CUDA 12.8 (same torch build as Task 1) plus the notebook's dependencies.
# Build:  docker build -t pytorch-task3 task3_gan/member_anushka/src
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# torch.compile needs a C compiler at run time to build its fused GPU kernels.
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir scipy pandas matplotlib nbconvert nbclient ipykernel lpips pytorch-fid
# Evaluation-only networks (FID InceptionV3, LPIPS AlexNet, course-script InceptionV3) are cached
# at build time so a run needs no network access. They are never used for training or generation.
RUN python -c "import lpips, torchvision; from pytorch_fid.inception import InceptionV3; InceptionV3([3]); lpips.LPIPS(net='alex'); torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)"
