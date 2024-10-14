# Use official Python image from DockerHub with GPU support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Set timezone to Asia/Kolkata
ENV TZ=Asia/Kolkata

# Install basic utilities and tzdata
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libopencv-dev \
    tzdata \
    ffmpeg \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean

# Install YOLOv5 dependencies, Torch, Torchvision, and TorchReID
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install additional Python dependencies for object tracking
RUN pip3 install \
    numpy \
    scipy \
    opencv-python-headless \
    munkres \
    pillow \
    torchreid \
    yolov5==6.0  # Adjust YOLOv5 version if needed

# Copy the project code into the container
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Expose a port if needed (if you're running a web app)
# EXPOSE 5000

# Set the entrypoint to execute the Python script
ENTRYPOINT ["python3", "sampledb2.py"]
