# Use NVIDIA's PyTorch image which comes with CUDA, cuDNN, and PyTorch pre-installed.
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set up virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

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

# Upgrade pip in venv
RUN pip install --upgrade pip setuptools wheel packaging

# Clone and install dlib from source to ensure it's compiled with CUDA support
# This will now install into the venv
RUN git clone https://github.com/davisking/dlib.git && \
    cd dlib && \
    mkdir build && \
    cd build && \
    cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 && \
    cmake --build . --config Release -j$(nproc) && \
    cd .. && \
    python setup.py install

# Install InsightFace from source (Moved BEFORE requirements to cache it)
# We clone the repo and install the python package into the venv
RUN git clone https://github.com/deepinsight/insightface.git && \
    cd insightface/python-package && \
    pip install . && \
    cd ../.. && \
    rm -rf insightface

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# The command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

