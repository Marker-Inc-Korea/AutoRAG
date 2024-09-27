# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install necessary system dependencies
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

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /usr/src/app

# Install pip and setuptools
RUN pip install --upgrade pip setuptools setuptools-scm

# Copy project files
COPY . /usr/src/app

# Install project dependencies
RUN pip install -r requirements.txt

# Install pytest for running tests
RUN pip install pytest

# Run tests
RUN pytest

# Define entrypoint for the container
ENTRYPOINT ["python"]

# Default command to run the application
CMD ["-m", "autorag.cli"]
