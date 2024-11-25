FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install watchfiles
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
