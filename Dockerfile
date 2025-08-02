FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download models
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('punkt')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads data/processed data/models data/cache logs static

# Set permissions
RUN chmod +x scripts/*.py

# Create non-root user
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "app/main.py"]
