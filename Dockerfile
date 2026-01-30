FROM python:3.10.11-slim

# System dependencies for OpenCV / ONNX / InsightFace
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        python-dotenv \
        python-multipart \
        jinja2

# Copy app code
COPY . .

# Railway-compatible startup
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
