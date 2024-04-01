cmake_minimum_required(VERSION 3.0.0)
# Define project name
project(demo)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
# SET("OpenCV_DIR" "/usr/local")

include_directories("include/")
# Declare the executable target built from your sources
add_executable(demo
               src/main.cpp
               src/detector.cpp
               src/utils.cpp
               src/nms.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
target_compile_features(demo PRIVATE cxx_std_14)

message(STATUS "========================================")
message(STATUS "               Onnx Runtime             ")
message(STATUS "========================================")
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