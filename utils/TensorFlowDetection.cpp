#include <memory>
#include <stdexcept>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <cv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

TensorFlowDetection::TensorFlowDetection()
{
    thresholdScore = 0.5;
    thresholdIOU = 0.8;
    labelPath = "demo/ssd_mobilenet_v1_egohands/labels_map.pbtxt";
    graphPath = tensorflow::io::JoinPath("../", "demo/ssd_mobilenet_v1_egohands/labels_map.pbtxt");
    inputLayer = "image_tensor:0";
    outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};
    LOG(INFO) << "graphPath:" << graphPath;
    tensorflow::Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok())
    {
        throw std::invalid_argument("loadGraph(): ERROR")
    } 
    else
        LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;
    
    labelsMap = std::map<int,std::string>();
    tensorflow::Status readLabelsMapStatus = readLabelsMapFile(tensorflow::io::JoinPath(ROOTDIR, LABELS), labelsMap);
    if (!readLabelsMapStatus.ok())
    {
        throw std::invalid_argument("readLabelsMapFile(): ERROR")
    }
    else
        LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << endl;

    shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim((int64)cap.get(CAP_PROP_FRAME_HEIGHT));
    shape.AddDim((int64)cap.get(CAP_PROP_FRAME_WIDTH));
    shape.AddDim(3);
}

tensorflow::Status TensorFlowDetection::loadGraph(const std::string &graph_file_name, std::unique_ptr<tensorflow::Session> *session)
{
    tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok())
    {
        return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
    }
    session->tensorlow::reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    tensorflow::Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return tensorflow::Status::OK();
}

tensorflow::Status TensorFlowDetection::readTensorFromMat(const cv::Mat &mat, tensorflow::Tensor &outTensor)
{

    auto root = tensorflow::Scope::NewRootScope();

    float *p = outTensor.flat<float>().data();
    cv::Mat fakeMat(mat.rows, mat.cols, cv::CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3);

    auto input_tensor = tensorflow::ops::Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    std::vector<pair<string, tensorflow::Tensor>> inputs = {{"input", outTensor}};
    auto uint8Caster = tensorflow::ops::Cast(root.WithOpName("uint8_Cast"), outTensor, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output outTensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::vector<Tensor> outTensors;
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"uint8_Cast"}, {}, &outTensors));

    outTensor = outTensors.at(0);
    return tensorflow::Status::OK();
}

cv::Mat TensorFlowDetection::singleDetection(const cv::Mat &frame)
{
    tensor = Tensor(tensorflow::DT_FLOAT, shape);
    tensorflow::Status readTensorStatus = readTensorFromMat(frame, tensor);
    if (!readTensorStatus.ok())
    {
        throw std::invalid_argument("Mat->Tensor conversion failed")
    }
    outputs.clear();
    if (!runStatus.ok())
    {
        throw std::invalid_argument("Running model failed")
    }
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
    tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float,3>();
    vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);
    for (size_t i = 0; i < goodIdxs.size(); i++)
            LOG(INFO) << "score:" << scores(goodIdxs.at(i)) << ",class:" << labelsMap[classes(goodIdxs.at(i))]
                      << " (" << classes(goodIdxs.at(i)) << "), box:" << "," << boxes(0, goodIdxs.at(i), 0) << ","
                      << boxes(0, goodIdxs.at(i), 1) << "," << boxes(0, goodIdxs.at(i), 2) << ","
                      << boxes(0, goodIdxs.at(i), 3);
    cv::cvtColor(frame, frame, COLOR_BGR2RGB);
    drawBoundingBoxesOnImage(frame, scores, classes, boxes, labelsMap, goodIdxs);
    return frame
}