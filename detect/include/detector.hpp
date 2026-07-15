#pragma once

#include <opencv2/opencv.hpp>
#include "config.hpp"

namespace rmdetect {

// ONNX 模型推理封装类
class Detector {
  public:
    explicit Detector(const Config &cfg);  // 构造函数：加载模型

    // 加载 ONNX 模型（构造函数自动调用）
    bool loadModel();

    // 对预处理后的 blob 做前向推理
    // 参数 blob: 预处理后的输入张量
    // 参数 output_shape: 返回输出张量的形状
    // 返回值: 展平后的输出数据
    std::vector<float> infer(const cv::Mat &blob,
                             std::vector<int> &output_shape);

    // 获取模型输入尺寸
    int inputWidth()  const { return cfg_.input_width;  }
    int inputHeight() const { return cfg_.input_height; }

  private:
    Config cfg_;                // 配置参数
    cv::dnn::Net net_;          // OpenCV DNN 网络对象
    bool loaded_ = false;       // 模型是否已成功加载
};

} // namespace rmdetect
