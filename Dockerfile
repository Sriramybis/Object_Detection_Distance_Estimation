# Base image with Python and OpenCV
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full application
COPY . .

# Expose Flask default port
EXPOSE 5050

# Run Flask app directly
CMD ["python", "-m", "app.app"]
