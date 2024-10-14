# Use a base image with Python 3.11 (or your preferred version)
FROM python:3.11.9

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (including libGL1 for OpenCV)
RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Set environment variable to prevent blocking stdout/stderr
ENV PYTHONUNBUFFERED=1

# Command to run your Python script
CMD ["python", "sampledb2.py"]
