/*================================================*/
/* Author: jason-li-831202                        */
/* @File: main.cpp                                */
/* @Software: Visual Stidio Code                  */
/*================================================*/

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
    cmd.add<std::string>("source", 's', "Source to be detected.", true, "video.mp4");
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", true, "coco.names");
    cmd.add<std::string>("score_thres", '\0', "Confidence threshold for categories.", false, "0.3f");
    cmd.add<std::string>("iou_thres", '\0', "Overlap threshold.", false, "0.4f");

    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    const bool isGPU = cmd.exist("gpu");
    const std::string sourcePath = cmd.get<std::string>("source");
    const std::string modelPath = cmd.get<std::string>("model_path");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const float confThreshold = std::stof(cmd.get<std::string>("score_thres"));
    const float iouThreshold = std::stof(cmd.get<std::string>("iou_thres"));

    bool isImage = utils::isImage(sourcePath);
    std::string outputPath = "";
    if (isImage)
    {
        outputPath = utils::splitExtension(sourcePath) + "_result.jpg";
    }
    else
    {
        outputPath = utils::splitExtension(sourcePath) + "_result.mp4";
    }

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

    if (!isImage)
    {

        cv::VideoCapture cap;

        // Check if source is webcam
        if (sourcePath == "0")
        {
            cap = cv::VideoCapture(0);
        }
        else
        {
            cap = cv::VideoCapture(sourcePath);
        }
        
        if (!cap.isOpened()) {
            LOG(ERROR) << "Cannot open video.\n";
            return -1;
        }
	    cv::Size S = cv::Size((int)cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH), 
                      (int)cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT));
        int fps = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);
        LOG(INFO) << "Current FPS : " << fps;
        cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, S, true);

        // TODO : add fps label
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
            if (cv::waitKey(1) == 'q') {
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

