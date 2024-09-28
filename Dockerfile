# Base stage: Install dependencies
FROM python:3.10-slim AS base

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc \
    libssl-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-kor && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /usr/src/app

# Copy project files
COPY . /usr/src/app

# Install Python dependencies
RUN pip install --upgrade pip setuptools setuptools-scm
RUN pip install -r requirements.txt

# Test stage: Run tests if CI=true
FROM base AS test

# Install testing dependencies
RUN pip install pytest pytest-xdist

# Run tests if CI is set to true
RUN pytest -o log_cli=true --log-cli-level=INFO -n auto tests

# Production stage: Create final image for production
FROM base AS production

COPY benchmark /usr/src/app/benchmark
COPY projects /usr/src/app/projects
COPY sample_config /usr/src/app/sample_config
COPY sample_dataset /usr/src/app/sample_dataset

# Set the entrypoint for the production application
ENTRYPOINT ["python", "-m", "autorag.cli"]
# ENTRYPOINT ["bash"]
