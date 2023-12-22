#!/bin/bash
CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)

ONNXRUNTIME_VERSION="1.15.0"
ONNXRUNTIME_GPU=1

# Platform
platform="$(uname -s)"
case "$platform" in
    Darwin*)
        ONNXRUNTIME_PLATFORM="osx"
        ONNXRUNTIME_GPU=0
        ;;
    Linux*)
        ONNXRUNTIME_PLATFORM="linux"
        ;;
    MINGW32_NT*|MINGW64_NT*)
        ONNXRUNTIME_PLATFORM="win"
        ;;
    *)
        echo "Unsupported platform: $platform"
        exit 1
        ;;
esac

# Architecture
architecture="$(uname -m)"

case "$architecture" in
    x86_64)
        ONNXRUNTIME_ARCH="x64"
        ;;
    armv7l)
        ONNXRUNTIME_ARCH="arm"
        ;;
    aarch64|arm64)
        ONNXRUNTIME_ARCH="arm64"
        ;;
    *)
        echo "Unsupported architecture: $architecture"
        exit 1
        ;;
esac


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
