#include <opencv2/opencv.hpp>
#include "utils.h"


namespace nms
{
    static bool sort_score(Detection bbox1, Detection bbox2);
    static float calc_iou(cv::Rect rect1, cv::Rect rect2);

    void nms_boxes(std::vector<cv::Rect> &boxes, 
                   std::vector<float> confidences_, 
                   float confthreshold, 
                   float nmsthreshold, 
                   std::vector<int> &indices);

    void soft_nms_boxes(std::vector<cv::Rect> &boxes, 
                        std::vector<float> confidences_, 
                        float confthreshold, 
                        float nmsthreshold, 
                        std::vector<int> &indices,
                        float sigma=0.5);

}