# Use Python 3.10 as base image
FROM python:3.10-slim

# Set metadata
LABEL maintainer="J Ko and Tim Ng"
LABEL description="VIs to LAI crops - Simulate Leaf Area Index from Vegetation Indices"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for scientific computing libraries
# Including cartopy dependencies for 2D mapping
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libhdf5-dev \
    libgomp1 \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and setup files first for better caching
COPY setup.py ./

# Install Python dependencies
# Install base requirements first, then the package in editable mode with all extras (dev + 2d)
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[all]"

# Copy all project files
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p outputs models data codes/each_crop_model \
    class_map_maize_Iowa class_map_rice_Paju \
    vis_maize_Iowa vis_rice_Paju \
    Shape_Paju_SK USA_Iowa_Map

# Expose Jupyter notebook port
EXPOSE 8888

# Default command - can be overridden when running the container
# Use Jupyter Lab as default for notebook development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]

