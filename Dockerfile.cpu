FROM ubuntu:22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user and set up directories
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app/outputs /app && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser requirements.txt ./requirements.txt

# Create and activate virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install CPU-only PyTorch and other dependencies
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    USE_GPU=false

# Expose the port
EXPOSE 5005

# Run FastAPI server with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "1"]
