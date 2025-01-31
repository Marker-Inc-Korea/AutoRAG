FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel as base

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc \
    libssl-dev

# Set work directory
WORKDIR /usr/src/app

# Install Python dependencies
RUN pip install --upgrade pip setuptools setuptools-scm

# Copy project files
COPY . /usr/src/app
RUN pip install -e ".[gpu]"

FROM base AS gpu
ENTRYPOINT ["python", "-m", "autorag.cli"]

FROM base AS gpu-parsing
RUN apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-kor
RUN pip install "AutoRAG[parse]"
ENTRYPOINT ["python", "-m", "autorag.cli"]
