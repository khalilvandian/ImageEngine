# Use NVIDIA's PyTorch image which comes with CUDA, cuDNN, and PyTorch pre-installed.
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for dlib and other libraries
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone and install dlib from source to ensure it's compiled with CUDA support
RUN git clone https://github.com/davisking/dlib.git && \
    cd dlib && \
    mkdir build && \
    cd build && \
    cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 && \
    cmake --build . && \
    cd .. && \
    python setup.py install

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port Marimo runs on (default is 2718, but can be changed)
EXPOSE 2718

# The command to run the Marimo application
CMD ["marimo", "run", "app.py", "--host", "0.0.0.0"]
