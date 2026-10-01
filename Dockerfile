# Text-to-motion server: the planner(s) on vLLM plus the motion generator, one REST API on port 8000.
#
#   docker build -t reachy-motion .
#   docker run --gpus all --ipc=host -p 8000:8000 -v reachy-motion-cache:/root/.cache reachy-motion
#
# Arguments after the image name go to inference.server (default: the 4B planner). For an older NVIDIA driver,
# build on the CUDA 12.9 variant: --build-arg VLLM_TAG=v0.30.0-cu129
ARG VLLM_TAG=v0.30.0
FROM vllm/vllm-openai:${VLLM_TAG}

# The base image has vLLM, PyTorch and CUDA; add only what it lacks, so none of its pinned libraries get replaced.
RUN pip install --no-cache-dir scipy reachy-mini-rust-kinematics
WORKDIR /app
COPY common common
COPY planner planner
COPY generator generator
COPY inference inference

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "inference.server"]
CMD ["--bundle", "medium=binhpham/reachy-mini-motion-planner-4b"]
