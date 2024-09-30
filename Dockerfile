FROM nvcr.io/nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /usr/src/app
COPY pyproject.toml uv.lock /usr/src/app/
RUN uv sync --frozen
ENV PATH="/usr/src/app/.venv/bin:$PATH"
COPY train.py /usr/src/app/
CMD ["python", "train.py", "--train-steps", "1000", "--eval-every", "100", "--batch-size", "32", "--learning-rate", "0.01", "--momentum", "0.9", "--output", "./output_path"]