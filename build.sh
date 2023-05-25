CURRENT_DIR=$(cd $(dirname $0); pwd)
ONNXRUNTIME_DIR="${CURRENT_DIR}/onnxruntime-linux-x64-gpu-1.9.0"

if [ -d "${CURRENT_DIR}/build" ]; then
    rm -rf build
    mkdir build 
    echo "Directory ${CURRENT_DIR}/build exists."
else
    mkdir build 
    echo "Directory ${CURRENT_DIR}/build does not exists."
fi

cd build
echo "Build Code ..."
cmake .. -D ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build .