FROM python:3.10-slim

WORKDIR /app

# Install system dependencies + parsing dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    libmagic-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-kor \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install watchfiles pdf2image
# Install NLTK and download model
RUN pip install nltk && \
    python3 -c "import nltk; nltk.download('punkt_tab')" && \
    python3 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# # Copy application code
# COPY . .

# Create directory for celery beat schedule
RUN mkdir -p /app/celerybeat

# Add entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    sed -i 's/\r$//' /entrypoint.sh  # Remove Windows line endings

ENTRYPOINT ["/entrypoint.sh"]
