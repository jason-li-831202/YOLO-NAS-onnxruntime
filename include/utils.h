#pragma once
#include <codecvt>
#include <fstream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "log.h"

struct Detection
{
    cv::Rect box;
    float conf{};
    int classId{};
};

typedef struct RGB {
    double r;
    double g;
    double b;
} RGB1;

class Timer {
    public:
        Timer(double& accumulator, bool isEnabled = true);
        void Stop();

    private:
        double& accumulator;
        bool isEnabled;
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

namespace utils
{
    bool isImage(const std::string& path);
    std::vector<cv::Scalar> colorVectorScalar(int num);
    std::string splitExtension(std::string fileNamePath);
    std::string getFileExtension(const std::string& path);
    size_t vectorProduct(const std::vector<int64_t>& vector);
    std::vector<float> arrayToVector(const float *data, std::vector<int64_t> outputShape);
    
    std::wstring charToWstring(const char* str);
    std::vector<std::string> loadNames(const std::string& path);
    static void cornerRect(cv::Mat& image, cv::Rect bbox, cv::Scalar color, int t, int rt);
    void drawDetectOnFrame(cv::Mat& image, std::vector<Detection>& detections,
                            const std::vector<std::string>& classNames,
                            std::vector<cv::Scalar> classColors);

    void letterBox(const cv::Mat& image, cv::Mat& outImage,
                   const cv::Size& newShape,
                   const cv::Scalar& color,
                   bool auto_,
                   bool scaleFill,
                   bool scaleUp,
                   int stride);

    void scaleCoords(const cv::Size& imageShape, cv::Rect& box, const cv::Size& imageOriginalShape);
    template <typename T>
    T clip(const T& n, const T& lower, const T& upper);
}
