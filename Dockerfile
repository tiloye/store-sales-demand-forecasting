FROM astrocrpublic.azurecr.io/runtime:3.2-4-python-3.12-base

# Switch to root for setup
USER root

# Setup project virtual environment
COPY pyproject.toml .
COPY uv.lock ./
COPY src/ssdf/__init__.py src/ssdf/__init__.py
COPY README.md .
RUN uv sync --no-dev

# Setup airflow dependencies
COPY requirements.txt .
RUN /usr/local/bin/install-python-dependencies

# Set environment variables
ENV PYTHON_ENVIRONMENT="/usr/local/airflow/.venv/bin/python"

# Switch back to astro user
USER astro

# Copy project files into image
COPY --chown=astro:0 . .
