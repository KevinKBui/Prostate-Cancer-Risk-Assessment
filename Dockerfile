FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY Data/ ./Data/

# Create necessary directories
RUN mkdir -p logs monitoring

# Expose port
EXPOSE 9696

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=src/web_service.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9696/health || exit 1

# Run the web service
CMD ["python", "src/web_service.py"]