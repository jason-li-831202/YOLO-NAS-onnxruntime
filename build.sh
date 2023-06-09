#!/bin/bash

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
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .