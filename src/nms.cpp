#include "nms.h"


static bool nms::sort_score(Detection bbox1, Detection bbox2)
{
    return bbox1.conf > bbox2.conf ? true : false;
}

static float nms::calc_iou(cv::Rect rect1, cv::Rect rect2)
{
    int xx1, yy1, xx2, yy2;
    xx1 = std::max(rect1.x, rect2.x);
    yy1 = std::max(rect1.y, rect2.y);
    xx2 = std::min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
    yy2 = std::min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

    int inter_width = std::max(0, xx2 - xx1 + 1);
    int inter_height = std::max(0, yy2 - yy1 + 1);

    float inter_area = float(inter_width) * inter_height;
    float union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - inter_area);
    float iou = inter_area / union_area;
    return iou;
}


void nms::nms_boxes(std::vector<cv::Rect> &boxes, std::vector<float> confidences_, float confthreshold, float nmsthreshold, std::vector<int> &indices)
{
    Detection bbox;
    std::vector<Detection> bboxes;
    int i, j;
    for (i = 0; i < boxes.size(); i++)
    {
        bbox.box = boxes[i];
        bbox.conf = confidences_[i];
        bbox.classId = i;
        bboxes.push_back(bbox);
    }
    std::sort(bboxes.begin(), bboxes.end(), sort_score);
    int k = bboxes.size();

    for (i = 0; i < k; i++)
    {
        if (bboxes[i].conf < confthreshold)
            continue;
        indices.push_back(bboxes[i].classId);
        for (j = i + 1; j < k; j++)
        {
            float iou = calc_iou(bboxes[i].box, bboxes[j].box);
            if (iou > nmsthreshold)
            {
                bboxes.erase(bboxes.begin() + j);
                k = bboxes.size();
                j--; // 修正這裡，將 j 減 1
            }
        }
    }

    // for (int i = 0; i < bboxes.size(); i++)
    // {
    //     std::cout << "bboxes:" << bboxes[i].box << std::endl;
    // }
}

void nms::soft_nms_boxes(std::vector<cv::Rect> &boxes, std::vector<float> confidences_, float confthreshold, float nmsthreshold, std::vector<int> &indices, float sigma)
{
    std::vector<Detection> bboxes;
    for (int i = 0; i < boxes.size(); i++)
    {
        Detection bbox;
        bbox.box = boxes[i];
        bbox.conf = confidences_[i];
        bbox.classId = i;
        bboxes.push_back(bbox);
    }
    std::sort(bboxes.begin(), bboxes.end(), sort_score);

    std::vector<int> temp_indices; // Temporary indices to avoid duplicates

    for (int i = 0; i < bboxes.size(); i++)
    {
        if (bboxes[i].conf < confthreshold)
            continue;
        
        // Find the index of the maximum score
        auto max_it = std::max_element(bboxes.begin() + i + 1, bboxes.end(),
                                       [](const Detection& a, const Detection& b) { return a.conf < b.conf; });
        int max_index = std::distance(bboxes.begin(), max_it);

        if (bboxes[i].conf < bboxes[max_index].conf)
        {
            std::swap(bboxes[i], bboxes[max_index]);
        }

        temp_indices.push_back(i); // Use index i instead of classId
        for (int j = i + 1; j < bboxes.size(); j++)
        {
            float iou = calc_iou(bboxes[i].box, bboxes[j].box);
            float weight = std::exp(-(iou * iou) / sigma);
            bboxes[j].conf *= weight;
        }
    }

    // Thresholding after applying soft-NMS
    for (int i = 0; i < temp_indices.size(); i++)
    {
        if (bboxes[temp_indices[i]].conf >= confthreshold)
        {
            indices.push_back(bboxes[temp_indices[i]].classId);
        }
    }
}