#ifndef TF_DETEC_H
#define TF_DETEC_H
#endif

#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include <opencv2/core/mat.hpp>

class TensorFlowDetection
{
    private:

        double thresholdScore;
        double thresholdIOU;

        std::string labelPath;
        std::string graphPath;
        std::string inputLayer;
        std::vector<std::string> outputLayer;
        std::unique_ptr<tensorflow::Session> session;
        std::map<int, std::string> labelsMap
        std::vector<tensorflow::Tensor> outputs
        
        tensorflow::Tensor tensor;
        tensorflow::TensorShape shape;

        tensorflow::Status loadGraph(const std::string &graph_file_name, std::unique_ptr<tensorflow::Session> *session);
        tensorflow::Status readTensorFromMat(const cv::Mat &mat, tensorflow::Tensor &outTensor);
    public:
        TensorFlowDetection();
        cv::Mat TensorFlowDetection::singleDetection(const cv::Mat &mat);
}