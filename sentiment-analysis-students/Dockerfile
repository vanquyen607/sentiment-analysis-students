<<<<<<< HEAD
# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Copy models directory
COPY tuned_models/ ./tuned_models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application

=======
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY static/ ./static/
COPY tuned_models/ ./tuned_models/

# Create static directory if not exists
RUN mkdir -p static

EXPOSE 8000

>>>>>>> 814168aca9c34789f814d891e2f3e095ec2a8468
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]