#include <iostream>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"

structlog LOGCFG = {};

int main(int argc, char* argv[])
{
    LOGCFG.headers = true; 
    LOGCFG.level = INFO;

    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to onnx model.", true, "yolov5.onnx");
    cmd.add<std::string>("image", 'i', "Image source to be detected.", false, "bus.jpg");
    cmd.add<std::string>("video", 'v', "Video source to be detected.", false, "video.mp4");
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", true, "coco.names");
    cmd.add<std::string>("score_thres", '\0', "Confidence threshold for categories.", false, "0.3f");
    cmd.add<std::string>("iou_thres", '\0', "Overlap threshold.", false, "0.4f");

    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    bool isVideo = cmd.exist("video");
    std::string sourcePath = "";
    std::string outputPath = "";
    if (cmd.exist("video"))
    {
        sourcePath = cmd.get<std::string>("video");
        outputPath = utils::splitExtension(sourcePath);
        outputPath.append("_result.mp4");
    }
    else if  (cmd.exist("image"))
    {
        sourcePath = cmd.get<std::string>("image");
        outputPath = utils::splitExtension(sourcePath);
        outputPath.append("_result.jpg");
        isVideo = false;
    }
    else
    {
        LOG(ERROR) << "Error: Empty Source Type, please check if there are parameters for video or image.";
        return -1;
    }

    const std::string modelPath = cmd.get<std::string>("model_path");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const float confThreshold = std::stof(cmd.get<std::string>("score_thres"));
    const float iouThreshold = std::stof(cmd.get<std::string>("iou_thres"));

    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    if (classNames.empty())
    {
        LOG(ERROR) << "Error: Empty class names file.";
        return -1;
    }

    YOLODetector detector {nullptr};
    cv::Mat image;
    std::vector<Detection> result;

    try
    {
        detector = YOLODetector(modelPath, isGPU);
        LOG(INFO) << "Model was initialized.";
    }
    catch(const std::exception& e)
    {
        LOG(ERROR) << e.what();
        return -1;
    }
    std::vector<cv::Scalar> classColors = utils::colorVectorScalar(detector.num_class);

    if (isVideo)
    {
        cv::VideoCapture cap(sourcePath);
        if (!cap.isOpened()) {
            LOG(ERROR) << "Cannot open camera\n";
            return -1;
        }
	    cv::Size S = cv::Size((int)cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH), 
                      (int)cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT));
        int fps = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);
        LOG(INFO) << "Current FPS : " << fps;
        cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, S, true);

        while (true) {
            bool ret = cap.read(image); 
            if (!ret) {
                LOG(WARN) << "Can't receive frame (stream end?). Exiting ...\n";
                break;
            }
            result = detector.detectFrame(image, confThreshold, iouThreshold);

            utils::drawDetectOnFrame(image, result, classNames, classColors);

            cv::imshow("result", image);
            writer.write(image);
            if (cv::waitKey(33) == 'q') {
                break;
            }
        }
        cap.release();
        writer.release();
    }
    else
    {
        image = cv::imread(sourcePath);

        result = detector.detectFrame(image, confThreshold, iouThreshold);

        utils::drawDetectOnFrame(image, result, classNames, classColors);

        cv::imshow("result", image);

        cv::imwrite(outputPath, image);
        cv::waitKey(0);
    }
    return 0;
}

