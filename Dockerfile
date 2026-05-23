FROM astrocrpublic.azurecr.io/runtime:3.2-4-python-3.12-base

# Switch to root for setup
USER root

# Install python dependencies
COPY pyproject.toml .
COPY uv.lock .
COPY src/ssdf/__init__.py src/ssdf/__init__.py
COPY README.md .
RUN uv sync --no-dev

# Set environment variables
ENV PYTHON_ENVIRONMENT="/usr/local/airflow/.venv/bin/python"

# Switch back to astro user
USER astro

# Copy project files into image
COPY --chown=astro:0 . .
