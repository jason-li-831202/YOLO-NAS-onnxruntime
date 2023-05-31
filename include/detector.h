#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>

#include "utils.h"


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

    static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                 float& bestConf, int& bestClassId);

};