# Use NVIDIA PyTorch base image for CUDA support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y 
    git 
    curl 
    libgl1-mesa-glx 
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements and install
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python3", "run.py", "--help"]
