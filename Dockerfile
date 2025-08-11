# Use Python 3.11 slim bookworm for better compatibility
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies
# ffmpeg is required for audio processing in Pipecat
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only torch (much smaller, no CUDA/NVIDIA dependencies)
# This saves ~2GB of image size and is all we need for Fly.io
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy the install_deps.py script for model pre-caching
COPY install_deps.py .

# Pre-cache the Silero VAD model during build to avoid runtime delays
# This significantly reduces cold start times
RUN python install_deps.py

# Copy the rest of the application
COPY . .

# Use PORT environment variable with default fallback to 8000
ENV PORT=8000

# Expose the port (informational)
EXPOSE 8000

# Run the FastAPI application
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]