# ─── Stage 1: builder ──-
FROM python:3.13-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LIBTORCH_URL=https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
ENV PROTOBUF_VERSION=3.13.0

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    unzip \
    build-essential \
    cmake \
    dos2unix \
    bash \
    cmake \
    make \
    libtorch-dev \
    && rm -rf /var/lib/apt/lists/*

# install protobuf
WORKDIR /tmp
RUN curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-cpp-3.13.0.tar.gz && \
    tar -xzf protobuf-cpp-${PROTOBUF_VERSION}.tar.gz && \
    cd protobuf-${PROTOBUF_VERSION} && \
    ./configure && make -j$(nproc) && make install && ldconfig

# Copy the needed folders for inference only
RUN mkdir -p /build

# Set working directory to MIDI-GPT
WORKDIR /build

COPY pip_requirements pip_requirements
COPY libraries libraries

COPY src src
COPY include include
COPY CMakeLists.txt /build/CMakeLists.txt
COPY models models
RUN unzip models/model.zip -d models


# Install Python dependencies
RUN pip install --no-cache-dir -r pip_requirements/common_requirements.txt

# Download and unpack libtorch
RUN mkdir -p libraries/libtorch && \
    curl -L ${LIBTORCH_URL} -o libtorch.zip && \
    unzip -q libtorch.zip -d libraries/ && \
    rm libtorch.zip

# Clone pybind11
RUN git clone https://github.com/pybind/pybind11.git libraries/pybind11 && \
    cd libraries/pybind11 && \
    git reset --hard 5ccb9e4

# Clone midifile
RUN git clone https://github.com/craigsapp/midifile libraries/midifile && \
    cd libraries/midifile && \
    git reset --hard 838c62c

# Compile protobuf definitions
RUN mkdir -p libraries/protobuf/build && \
    protoc --proto_path=libraries/protobuf/src --cpp_out=libraries/protobuf/build libraries/protobuf/src/*.proto

RUN cp -r /build/libraries/libtorch /opt/
ENV TORCH_DIR=/opt/libtorch
ENV LD_LIBRARY_PATH=$TORCH_DIR/lib:$LD_LIBRARY_PATH

# Build the C++ Python extension
RUN mkdir -p python_lib && \
    cd python_lib && \
    cmake .. -DCMAKE_PREFIX_PATH=$TORCH_DIR && \
    make


WORKDIR /app
# Copy the compiled C++ extension and models from the builder stage

RUN cp -r /build/python_lib python_lib
RUN cp -r /build/models models

ENV PYTHONPATH=/app/python_lib
RUN rm -rf /build
RUN python3 -c "import midigpt; print('✅ midigpt built and importable')"


CMD ["/bin/bash"]