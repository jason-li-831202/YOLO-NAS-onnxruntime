#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>

#include "utils.h"
#include "nms.h"

static uint16_t float32_to_float16(float& input_fp32);
static float float16_to_float32(uint16_t& input_fp16);

inline const char* ToString(ONNXTensorElementDataType v)
{
    switch (v)
    {
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT   : return "FLOAT" ;      // maps to c type float
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8   : return "UINT8" ;      // maps to c type uint8_t
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8    : return "INT8"  ;      // maps to c type int8_t
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16  : return "UINT16";      // maps to c type uint16_t
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16   : return "INT16" ;      // maps to c type int16_t
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32   : return "INT32" ;      // maps to c type int32_t
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64   : return "INT64" ;      // maps to c type int64_t
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING  : return "STRING" ;     // maps to c++ type std::string
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL    : return "BOOL" ;  
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : return "FLOAT16" ;  
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE  : return "DOUBLE" ;      // maps to c type double
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32  : return "UINT32" ;      // maps to c type uint32_t
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64  : return "UINT64" ;      // maps to c type uint64_t
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return "COMPLEX64" ;  // complex with float32 real and imaginary components
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return "COMPLEX128"; // complex with float64 real and imaginary components
        case  ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "BFLOAT16" ;    // Non-IEEE floating-point format based on IEEE754 single-precision
        default:      return "UNDEFINED";
    }
}

class YOLODetector
{
public:
    explicit YOLODetector(std::nullptr_t) {};
    YOLODetector(const std::string& modelPath, const bool& isGPU);
    std::vector<Detection> detectFrame(cv::Mat &image, const float& confThreshold, const float& iouThreshold);
	int num_class;
	int num_proposal;

private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};
    ONNXTensorElementDataType inputType;
    ONNXTensorElementDataType outputType;
    bool isDynamicInputShape{};
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    #if ORT_API_VERSION >= 13
        std::vector<std::string> inputNamesString;
        std::vector<std::string> outputNamesString;
    #endif
	std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
    cv::Size2f inputImageShape;

    void getInputDetails(Ort::AllocatorWithDefaultOptions allocator);
    void getOutputDetails(Ort::AllocatorWithDefaultOptions allocator);
    void preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape);
    std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
                                          const cv::Size& originalImageShape,
                                          std::vector<Ort::Value>& outputTensors,
                                          const float& confThreshold, const float& iouThreshold);

    static void getBestClassInfo(float* it, const int& numClasses,
                                 float& bestConf, int& bestClassId); 

};