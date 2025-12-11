# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements_api.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for data and models if they don't exist
RUN mkdir -p /app/data/01-raw /app/data/02-preprocessed /app/models /app/mlruns

# Expose port for FastAPI application (default: 8000)
EXPOSE 8000

# Expose port for Gradio interface (default: 7860)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command: run FastAPI application with Uvicorn
CMD ["uvicorn", "src.api_app:app", "--host", "0.0.0.0", "--port", "8000"]
