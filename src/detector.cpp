#include "detector.h"

YOLODetector::YOLODetector(const std::string& modelPath,
                           const bool& isGPU = true)
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    LOG(INFO) << "=============== Model info ===============";
    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        LOG(WARN) << "GPU is not supported by your ONNXRuntime build. Fallback to CPU.";
        LOG(INFO) << "Inference device: CPU";
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        LOG(INFO) << "Inference device: GPU";
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        LOG(INFO) << "Inference device: CPU";
    }

    LOG(INFO) << "Inference model: " << modelPath;
#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    this->getInputDetails(allocator);
    this->getOutputDetails(allocator);

    cv::Size inputSize = cv::Size(input_node_dims[0][3], input_node_dims[0][2]);
    this->inputImageShape = cv::Size2f(inputSize);
	this->num_proposal = output_node_dims[0][1];
	this->num_class = output_node_dims[0][2]-5; // first 5 elements are box[4] and obj confidence
    LOG(INFO) << "Class num: " << this->num_class;
    LOG(INFO) << "==========================================";
}

void YOLODetector::getInputDetails(Ort::AllocatorWithDefaultOptions allocator)
{
    LOG(INFO) << "---------------- Input info --------------";
    this->isDynamicInputShape = false;
    for (int layer=0; layer < this->session.GetInputCount(); layer+=1)
    {
        inputNames.push_back(this->session.GetInputName(layer, allocator));
        LOG(INFO) << "Name [" << layer << "]: " << inputNames[layer];

        std::vector<int64_t> inputTensorShape = this->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        // checking if width and height are dynamic
        if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
        {
            LOG(INFO) << "Dynamic input shape.";
            this->isDynamicInputShape = true;
        }
        input_node_dims.push_back(inputTensorShape);
        LOG(INFO, true, false) << "Shape [" << layer << "]: (" << "";
        for (const int64_t& shape : inputTensorShape)
            LOG(INFO, false, false) << shape << ", ";
        LOG(INFO, false) << ")";
    }
}

void YOLODetector::getOutputDetails(Ort::AllocatorWithDefaultOptions allocator)
{
    LOG(INFO) << "--------------- Output info --------------";
    for (int layer=0; layer < this->session.GetOutputCount(); layer+=1)
    {
        outputNames.push_back(this->session.GetOutputName(layer, allocator));
        LOG(INFO) << "Name [" << layer << "]: " << outputNames[layer];
        
        auto outputTensorShape = this->session.GetOutputTypeInfo(layer).GetTensorTypeAndShapeInfo().GetShape();
        output_node_dims.push_back(outputTensorShape);
        LOG(INFO, true, false) << "Shape [" << layer << "]: (" << "";
        for (const int64_t& shape : outputTensorShape)
            LOG(INFO, false, false) << shape << ", ";
        LOG(INFO, false) << ")";
    }
}

void YOLODetector::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                    float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}

void YOLODetector::preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterBox(resizedImage, resizedImage, this->inputImageShape,
                     cv::Scalar(114, 114, 114), this->isDynamicInputShape,
                     false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    LOG(DEBUG) << "ori image shape (h, w):  "<< image.rows << " " << image.cols;
    LOG(DEBUG) << "resize image shape (h, w):  "<< resizedImage.rows << " " << resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize {floatImage.cols, floatImage.rows};

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> YOLODetector::postprocessing(const cv::Size& resizedImageShape,
                                                    const cv::Size& originalImageShape,
                                                    std::vector<Ort::Value>& outputTensors,
                                                    const float& confThreshold, const float& iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    for (int layer=0; layer < output_node_dims.size(); layer+=1)
    {
        std::vector<int64_t> outputShape = output_node_dims[layer];
        const float * rawOutput = outputTensors[layer].GetTensorData<float>();
        size_t count = outputTensors[layer].GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> output(rawOutput, rawOutput + count);
        int elementsInBatch = (int)(output_node_dims[layer][1] * output_node_dims[layer][2]); 
        LOG(DEBUG) << "output size :" << output.size();
        LOG(DEBUG) << "elementsInBatch :" << elementsInBatch;

        // only for batch size = 1
        for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += output_node_dims[layer][2])
        {
            float clsConf = it[4];
            if (clsConf > confThreshold)
            {
                int left = (int) (it[0]);
                int top = (int) (it[1]);
                int right = (int) (it[2]);
                int bottom = (int) (it[3]);
                int width = (int) right - left;
                int height = (int)  bottom - top;

                float objConf;
                int classId;
                this->getBestClassInfo(it, this->num_class, objConf, classId);

                float confidence = clsConf * objConf;

                boxes.emplace_back(left, top, width, height);
                confs.emplace_back(confidence);
                classIds.emplace_back(classId);
            }
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    LOG(DEBUG) << "amount of NMS indices: " << indices.size();

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        utils::scaleCoords(resizedImageShape, det.box, originalImageShape);

        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}

std::vector<Detection> YOLODetector::detectFrame(cv::Mat &image, const float& confThreshold = 0.4,
                                            const float& iouThreshold = 0.45)
{
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape {1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size()
    ));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              inputNames.data(),
                                                              inputTensors.data(),
                                                              inputNames.size(),
                                                              outputNames.data(),
                                                              outputNames.size() );

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<Detection> result = this->postprocessing(resizedShape,
                                                         image.size(),
                                                         outputTensors,
                                                         confThreshold, iouThreshold);

    delete[] blob;

    return result;
}
