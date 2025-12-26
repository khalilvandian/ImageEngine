# Use NVIDIA's PyTorch image which comes with CUDA, cuDNN, and PyTorch pre-installed.
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for dlib and other libraries
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    cmake \
    build-essential \
    fonts-liberation \
    && git lfs install --system \
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

# Expose the Streamlit default port
EXPOSE 8501

# The command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
