# Base stage: Install common dependencies
FROM python:3.10-slim AS base

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc \
    libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /usr/src/app

# Install Python dependencies
RUN pip install --upgrade pip setuptools setuptools-scm

# Copy project files
COPY ../.. /usr/src/app
RUN pip install -e ./ # Install project dependencies and AutoRAG as editable

# Ko stage
FROM base AS ko
RUN pip install "AutoRAG[ko]"
ENTRYPOINT ["python", "-m", "autorag.cli"]

# Dev stage
FROM base AS dev
RUN pip install -r ./tests/requirements.txt
RUN pip install -r ./docs/requirements.txt
RUN pip install "AutoRAG[dev]"
ENTRYPOINT ["python", "-m", "autorag.cli"]

# Ja stage
FROM base AS ja
RUN pip install "AutoRAG[ja]"
ENTRYPOINT ["python", "-m", "autorag.cli"]

FROM base AS api
RUN rm -rf ./sample_dataset
RUN rm -rf ./tests
RUN rm -rf ./docs
ENTRYPOINT ["python", "-m", "autorag.cli"]

# Parsing stage
FROM base AS parsing
RUN apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-kor
RUN pip install "AutoRAG[parse]"
ENTRYPOINT ["python", "-m", "autorag.cli"]
