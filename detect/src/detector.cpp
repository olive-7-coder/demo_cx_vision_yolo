#include "detector.hpp"
#include <iostream>

namespace rmdetect {

// 构造函数：保存配置并加载模型
Detector::Detector(const Config &cfg) : cfg_(cfg) {
    loadModel();
}

// 加载 ONNX 模型
bool Detector::loadModel() {
    try {
        net_ = cv::dnn::readNetFromONNX(cfg_.model_path);
    } catch (const cv::Exception &e) {
        std::cerr << "[检测器] 加载模型失败: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }

    if (net_.empty()) {
        std::cerr << "[检测器] 模型为空: " << cfg_.model_path << std::endl;
        loaded_ = false;
        return false;
    }

    if (cfg_.use_cuda) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "[检测器] 使用 CUDA 加速" << std::endl;
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::cout << "[检测器] 模型已加载: " << cfg_.model_path << std::endl;
    loaded_ = true;
    return true;
}

// 前向推理：输入预处理后的 blob，返回扁平化的输出数据
std::vector<float> Detector::infer(const cv::Mat &blob,
                                   std::vector<int> &output_shape) {
    if (!loaded_) {
        std::cerr << "[检测器] 模型未加载，无法推理" << std::endl;
        return {};
    }

    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    if (outputs.empty()) {
        std::cerr << "[检测器] 前向传播未返回结果" << std::endl;
        return {};
    }

    cv::Mat &out = outputs[0];

    int dims = out.dims;
    output_shape.clear();
    for (int d = 0; d < dims; ++d) {
        output_shape.push_back(out.size[d]);
    }

    size_t total = out.total();
    std::vector<float> result(total);
    if (out.isContinuous()) {
        memcpy(result.data(), out.ptr<float>(), total * sizeof(float));
    } else {
        for (size_t i = 0; i < total; ++i) {
            result[i] = out.at<float>(i);
        }
    }

    return result;
}

} // namespace rmdetect
