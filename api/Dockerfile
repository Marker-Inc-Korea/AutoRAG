FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and pip
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE ${PORT:-5000}
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${PORT:-5000}", "--reload", "--loop", "asyncio"]
