# Use Ubuntu as base image for better compatibility with whisper.cpp
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    cmake \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Clone whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git && \
    cd whisper.cpp && \
    make

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads output

# Add build argument for model name
ARG MODEL_NAME=large-v3-turbo

# Download the model
RUN cd whisper.cpp && \
    bash ./models/download-ggml-model.sh ${MODEL_NAME}

# Set environment variables
ENV WHISPER_PATH=/app/whisper.cpp
ENV WHISPER_CLI=/app/whisper.cpp/build/bin/whisper-cli
ENV MODEL_PATH=/app/whisper.cpp/models/ggml-${MODEL_NAME}.bin
ENV UPLOAD_DIR=/app/uploads
ENV OUTPUT_DIR=/app/output

# Expose port
EXPOSE 8000

# Run the application
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 