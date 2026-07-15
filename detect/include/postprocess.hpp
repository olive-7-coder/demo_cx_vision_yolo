#pragma once

#include <vector>
#include "config.hpp"
#include "preprocess.hpp"

namespace rmdetect {

// 解码 YOLO 模型的 ONNX 输出，提取检测结果
// 支持 3 维输出 [1, 4+num_classes, N]（YOLOv8/v11 格式）
// 参数 raw_output: 模型输出的原始数据
// 参数 output_shape: 输出张量的形状
// 参数 letterbox_info: letterbox 缩放信息（用于坐标回映射）
// 参数 cfg: 配置（类别数、阈值等）
// 返回值: 检测结果列表
std::vector<Detection> decodeOutput(
    const float        *raw_output,
    const std::vector<int> &output_shape,
    const LetterboxInfo    &letterbox_info,
    const Config           &cfg
);

// 非极大值抑制（NMS）去重
// 对重叠的检测框只保留置信度最高的一个
std::vector<Detection> nonMaxSuppression(
    std::vector<Detection> &detections,
    float nms_threshold
);

} // namespace rmdetect
