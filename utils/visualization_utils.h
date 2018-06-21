#ifndef VISUALISATION_UTILS_H
#define VISUALISATION_UTILS_H
#endif

#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include <opencv2/core/mat.hpp>

void drawBoundingBoxOnImage(cv::Mat &image, double xMin, double yMin, double xMax, double yMax, double score, std::string label, bool scaled);
void drawBoundingBoxesOnImage(cv::Mat &image,
                              tensorflow::TTypes<float>::Flat &scores,
                              tensorflow::TTypes<float>::Flat &classes,
                              tensorflow::TTypes<float,3>::Tensor &boxes,
                              std::map<int, string> &labelsMap,
                              std::vector<size_t> &idxs);
std::vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                                tensorflow::TTypes<float, 3>::Tensor &boxes,
                                double thresholdIOU, double thresholdScore);