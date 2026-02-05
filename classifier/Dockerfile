FROM python:3.12 

WORKDIR /app
RUN pip install --no-cache-dir uv

# Set env vars for optimization
ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1

# 1. Install Dependencies (with Caching!)
COPY pyproject.toml uv.lock ./

# usage of --mount=type=cache speeds up re-builds significantly
# cache all packages in a folder so they can be reused in future builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2. Copy Code & Install Project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Set the locale to UTF-8 (support mandarin chars)
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

# Expose the port that the server will run on
EXPOSE 8001
CMD ["uv", "run", "mcp_server.py"]