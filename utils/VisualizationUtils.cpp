#include <vector>
#include <cv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "VisualizationUtils.h"
#include "tensorflow/core/framework/tensor.h"

void drawBoundingBoxOnImage(cv::Mat &image, double yMin, double xMin, double yMax, double xMax, double score, std::string label, bool scaled=true)
{
    cv::Point tl, br;
    if (scaled)
    {
        tl = cv::Point((int) (xMin * image.cols), (int) (yMin * image.rows));
        br = cv::Point((int) (xMax * image.cols), (int) (yMax * image.rows));
    }
    else
    {
        tl = cv::Point((int) xMin, (int) yMin);
        br = cv::Point((int) xMax, (int) yMax);
    }

    cv::rectangle(image, tl, br, cv::Scalar(0, 255, 255), 1);

    float scoreRounded = floorf(score * 1000) / 1000;
    std::string scoreString = std::to_string(scoreRounded).substr(0, 5);
    std::string caption = label + " (" + scoreString + ")";

    int fontCoeff = 12;
    cv::Point brRect = cv::Point(tl.x + caption.length() * fontCoeff / 1.6, tl.y + fontCoeff);
    cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 255), -1);
    cv::Point textCorner = cv::Point(tl.x, tl.y + fontCoeff * 0.9);
    cv::putText(image, caption, textCorner, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
}

void drawBoundingBoxesOnImage(cv::Mat &image, tensorflow::TTypes<float>::Flat &scores, tensorflow::TTypes<float>::Flat &classes, tensorflow::TTypes<float,3>::Tensor &boxes, std::map<int, std::string> &labelsMap, std::vector<size_t> &idxs)
{
    for (int j = 0; j < idxs.size(); j++)
        drawBoundingBoxOnImage(image, boxes(0,idxs.at(j),0), boxes(0,idxs.at(j),1), boxes(0,idxs.at(j),2), boxes(0,idxs.at(j),3), scores(idxs.at(j)), labelsMap[classes(idxs.at(j))]);
}

double IOU(cv::Rect2f box1, cv::Rect2f box2)
{

    float xA = std::max(box1.tl().x, box2.tl().x);
    float yA = std::max(box1.tl().y, box2.tl().y);
    float xB = std::min(box1.br().x, box2.br().x);
    float yB = std::min(box1.br().y, box2.br().y);

    float intersectArea = std::abs((xB - xA) * (yB - yA));
    float unionArea = std::abs(box1.area()) + std::abs(box2.area()) - intersectArea;

    return 1. * intersectArea / unionArea;
}

std::vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores, tensorflow::TTypes<float, 3>::Tensor &boxes, double thresholdIOU, double thresholdScore)
{

    std::vector<size_t> sortIdxs(scores.size());
    std::iota(sortIdxs.begin(), sortIdxs.end(), 0);

    std::set<size_t> badIdxs = std::set<size_t>();
    size_t i = 0;
    while (i < sortIdxs.size())
    {
        if (scores(sortIdxs.at(i)) < thresholdScore)
            badIdxs.insert(sortIdxs[i]);
        if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end())
        {
            i++;
            continue;
        }

        cv::Rect2f box1 = cv::Rect2f(cv::Point2f(boxes(0, sortIdxs.at(i), 1), boxes(0, sortIdxs.at(i), 0)), cv::Point2f(boxes(0, sortIdxs.at(i), 3), boxes(0, sortIdxs.at(i), 2)));
        for (size_t j = i + 1; j < sortIdxs.size(); j++)
        {
            if (scores(sortIdxs.at(j)) < thresholdScore)
            {
                badIdxs.insert(sortIdxs[j]);
                continue;
            }
            cv::Rect2f box2 = cv::Rect2f(cv::Point2f(boxes(0, sortIdxs.at(j), 1), boxes(0, sortIdxs.at(j), 0)), cv::Point2f(boxes(0, sortIdxs.at(j), 3), boxes(0, sortIdxs.at(j), 2)));
            if (IOU(box1, box2) > thresholdIOU)
                badIdxs.insert(sortIdxs[j]);
        }
        i++;
    }

    std::vector<size_t> goodIdxs = std::vector<size_t>();
    for (auto it = sortIdxs.begin(); it != sortIdxs.end(); it++)
        if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end())
            goodIdxs.push_back(*it);

    return goodIdxs;
}