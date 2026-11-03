# CPU runtime. The selected combined profile uses the reconstruction target.
FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /app
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0'
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs
RUN python -m pip install --no-cache-dir '.[dev]'
CMD ["python", "scripts/generate_gt.py", "--synthetic", "--max-samples", "3"]

FROM runtime AS reconstruction
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgfortran5 libgl1 libegl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir '.[reconstruction,visualization,reports]'
RUN python -c "import open3d; print('Open3D runtime:', open3d.__version__)"

FROM runtime AS default
