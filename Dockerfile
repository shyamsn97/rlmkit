# Sandbox image for rlmkit's DockerRuntime.
#
# Build:
#   docker build -t rlmkit:local .
#
# Use:
#   from rlmkit.runtime.docker import DockerRuntime
#   runtime = DockerRuntime("rlmkit:local")
#
# Or via any of the bundled examples:
#   python examples/summarizer.py --runtime docker --docker-image rlmkit:local

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/rlmkit
COPY pyproject.toml README.md ./
COPY rlmkit ./rlmkit
RUN pip install ".[openai,anthropic]"

# DockerRuntime bind-mounts the host workspace at /workspace.
WORKDIR /workspace

# DockerRuntime spawns: `docker run -i --rm <image> python -m rlmkit.runtime.repl`.
# Setting it as CMD also makes `docker run -i rlmkit:local` work standalone.
CMD ["python", "-m", "rlmkit.runtime.repl"]
