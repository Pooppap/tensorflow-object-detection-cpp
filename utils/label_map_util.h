#ifndef LABEL_UTILS_H
#define LABEL_UTILS_H
#endif

#include "tensorflow/core/framework/tensor.h"

tensorflow::Status readLabelsMapFile(const std::string &fileName, std::map<int, std::string> &labelsMap);