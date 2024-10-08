# Base stage: Install common dependencies
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

# Install Python dependencies
RUN pip install --upgrade pip setuptools setuptools-scm
COPY requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

# Copy project files
COPY . /usr/src/app
RUN pip install -e ./

# Test stage
FROM base AS test
RUN pip install pytest pytest-xdist

# Ko stage
FROM base AS ko
RUN pip install --no-cache-dir kiwipiepy>=0.18.0 konlpy
ENTRYPOINT ["python", "-m", "autorag.cli"]

# Dev stage
FROM base AS dev
RUN pip install --no-cache-dir ruff pre-commit
ENTRYPOINT ["python", "-m", "autorag.cli"]

# Parsing stage
FROM base AS parsing
RUN pip install --no-cache-dir PyMuPDF pdfminer.six pdfplumber unstructured jq "unstructured[pdf]" "PyPDF2<3.0" pdf2image
ENTRYPOINT ["python", "-m", "autorag.cli"]

# Parsing stage
FROM base AS all
# TODO
ENTRYPOINT ["python", "-m", "autorag.cli"]

# Production stage (includes all features)
FROM base AS production
RUN pip install --no-cache-dir \
    kiwipiepy>=0.18.0 konlpy \
    ruff pre-commit \
    PyMuPDF pdfminer.six pdfplumber unstructured jq "unstructured[pdf]" "PyPDF2<3.0" pdf2image

COPY projects /usr/src/app/projects

ENTRYPOINT ["python", "-m", "autorag.cli"]
