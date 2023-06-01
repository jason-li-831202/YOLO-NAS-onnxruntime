#!/bin/bash
CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)

ONNXRUNTIME_VERSION="1.15.0"
ONNXRUNTIME_GPU=1

# Platform
if [ "$(uname)" == "Darwin" ]; then
    ONNXRUNTIME_PLATFORM="osx"
    ONNXRUNTIME_GPU=0
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    ONNXRUNTIME_PLATFORM="linux"
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    ONNXRUNTIME_PLATFORM="windows"
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
    ONNXRUNTIME_PLATFORM="windows"
else 
    echo "Unsupported platform"
    exit 1
fi

# Architecture
if [ "$(uname -m)" == "x86_64" ]; then
    ONNXRUNTIME_ARCH="x64"
elif [ "$(uname -m)" == "armv7l" ]; then
    ONNXRUNTIME_ARCH="arm"
elif [ "$(uname -m)" == "aarch64" ] || [ "$(uname -m)" == "arm64" ]; then
    ONNXRUNTIME_ARCH="arm64"
else
    echo "Unsupported $(uname -m) architecture"
    exit 1
fi

# GPU
if [ ${ONNXRUNTIME_GPU} == 1 ]; then
    ONNXRUNTIME_PATH="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}-gpu-${ONNXRUNTIME_VERSION}"
else
    ONNXRUNTIME_PATH="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}"
fi

# Download onnxruntime
if [ ! -d "${CURRENT_DIR}/${ONNXRUNTIME_PATH}" ]; then
    echo "Downloading onnxruntime ..." 
    curl -L -O -C - https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_PATH}.tgz
    tar -zxvf ${ONNXRUNTIME_PATH}.tgz
fi

ONNXRUNTIME_DIR="${CURRENT_DIR}/${ONNXRUNTIME_PATH}"

if [ -d "${CURRENT_DIR}/build" ]; then
    rm -rf build
    mkdir build
    echo "Directory ${CURRENT_DIR}/build exists."
else
    mkdir build 
    echo "Directory ${CURRENT_DIR}/build does not exist."
fi

cd build
echo "Build Code ..."
cmake .. -D ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build .
