# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Download the model during build time (to ensure it's included in the image)
ENV TRANSFORMERS_OFFLINE=0
ENV HF_DATASETS_OFFLINE=0
ENV HF_HUB_OFFLINE=0
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', device='cpu')"

# Set the default command
CMD ["python", "analyze_documents.py"]

# Metadata
LABEL \
    maintainer="TheBoys <kedareswar.pattapu@gmail.com>" \
    version="1.0.0" \
    description="Multi-Collection PDF Analyzer for Adobe India Hackathon 2025 - Challenge 1B"
