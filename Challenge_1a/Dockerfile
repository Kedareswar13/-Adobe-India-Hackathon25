# Use the official Python 3.10 slim image as the base
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set the default command
CMD ["python", "process_pdfs.py"]

# Metadata
LABEL \
    maintainer="TheBoys <kedareswar.pattapu@gmail.com>" \
    version="1.0.0" \
    description="PDF Outline Extractor for Adobe India Hackathon 2025 - Challenge 1A"
