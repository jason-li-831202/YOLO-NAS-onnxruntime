cmake_minimum_required(VERSION 3.0.0)

# Define project name
project(demo)

message(STATUS "========================================")
message(STATUS "               Onnx Runtime             ")
message(STATUS "========================================")

set(ONNXRUNTIME_VERSION "1.10.0")
set(ONNXRUNTIME_GPU 1)

# Platform
if(APPLE)
    set(ONNXRUNTIME_PLATFORM "osx")
    set(ONNXRUNTIME_GPU 0)
elseif(UNIX AND NOT APPLE)
    set(ONNXRUNTIME_PLATFORM "linux")
elseif(WIN32)
    set(ONNXRUNTIME_PLATFORM "win")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# Architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ONNXRUNTIME_ARCH "x64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l")
    set(ONNXRUNTIME_ARCH "arm")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ONNXRUNTIME_ARCH "arm64")
else()
    message(FATAL_ERROR "Unsupported architecture")
endif()

# GPU is available
if(${ONNXRUNTIME_GPU} EQUAL 1)
    set(ONNXRUNTIME_PATH "onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}-gpu-${ONNXRUNTIME_VERSION}")
else()
    set(ONNXRUNTIME_PATH "onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}")
endif()

# Download onnxruntime
set(CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT EXISTS "${CURRENT_DIR}/${ONNXRUNTIME_PATH}")
    message("Downloading onnxruntime ...")
    file(DOWNLOAD "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_PATH}.tgz"
        "${CURRENT_DIR}/${ONNXRUNTIME_PATH}.tgz"
        SHOW_PROGRESS)
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${CURRENT_DIR}/${ONNXRUNTIME_PATH}.tgz"
        WORKING_DIRECTORY ${CURRENT_DIR})
endif()

set(ONNXRUNTIME_DIR "${CURRENT_DIR}/${ONNXRUNTIME_PATH}")
# SET("OpenCV_DIR" "/usr/local")

include_directories("include/")

# Declare the executable target built from your sources
add_executable(demo
               src/main.cpp
               src/detector.cpp
               src/utils.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
target_compile_features(demo PRIVATE cxx_std_14)

find_library(PATH ${ONNXRUNTIME_DIR})
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}\n")

# Link your application with Onnxruntime libraries/header
target_include_directories(demo PRIVATE "${ONNXRUNTIME_DIR}/include")
if (WIN32)
    target_link_libraries(demo "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
elseif (APPLE)    
    target_link_libraries(demo "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
elseif (UNIX)
    target_link_libraries(demo "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)

message(STATUS "========================================")
message(STATUS "                OpenCV                  ")
message(STATUS "========================================")
# find_library(PATH ${OpenCV_DIR})
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Library Status:")
message(STATUS "   Version: ${OpenCV_VERSION}")
message(STATUS "   Libraries: ${OpenCV_LIBS}")
message(STATUS "   Include path: ${OpenCV_INCLUDE_DIRS}\n")

# Link your application with OpenCV libraries/header
target_include_directories(demo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(demo ${OpenCV_LIBS})