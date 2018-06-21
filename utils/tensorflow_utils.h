#ifndef TF_UTILS_H
#define TF_UTILS_H
#endif

#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include <opencv2/core/mat.hpp>

tensorflow::Status loadGraph(const std::string &graph_file_name, std::unique_ptr<tensorflow::Session> *session);
tensorflow::Status readTensorFromMat(const cv::Mat &mat, tensorflow::Tensor &outTensor);
double IOU(cv::Rect box1, cv::Rect box2);