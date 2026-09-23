# File: Dockerfile
# Purpose: Defines the container environment for the Python service.
# Connects: Used by docker-compose.yml to build the application image.
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY src/ src/

# Set the Python path to include the src directory so modules can be resolved
ENV PYTHONPATH=/app/src

# Default command to run the main application script
CMD ["python", "src/main.py"]
