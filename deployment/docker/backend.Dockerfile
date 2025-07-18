# Use Python 3.10 with CUDA support for face recognition
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libgphoto2-dev \
    libeigen3-dev \
    libhdf5-serial-dev \
    qtbase5-dev \
    libfaiss-dev \
    curl \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install additional face recognition dependencies
RUN pip install --no-cache-dir \
    onnxruntime-gpu \
    opencv-contrib-python \
    dlib

# Copy application code
COPY backend/ ./backend/
COPY database/ ./database/
COPY scripts/ ./scripts/

# Copy existing face recognition modules
COPY API_experimentation.py ./legacy/
COPY db_config.py ./legacy/
COPY db_manager.py ./legacy/
COPY db_models.py ./legacy/
COPY face_enroller.py ./legacy/
COPY enrollment_system_code.txt ./legacy/

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs /app/models

# Download InsightFace models (optional - can be done at runtime)
RUN python -c "
from insightface.app import FaceAnalysis
import os
os.makedirs('/root/.insightface', exist_ok=True)
try:
    app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
    print('InsightFace models downloaded successfully')
except Exception as e:
    print(f'Model download failed: {e}')
    print('Models will be downloaded at runtime')
"

# Set permissions
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --user-group facial_recognition

# Change ownership of app directory
RUN chown -R facial_recognition:facial_recognition /app

# Switch to non-root user for most operations
USER facial_recognition

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]