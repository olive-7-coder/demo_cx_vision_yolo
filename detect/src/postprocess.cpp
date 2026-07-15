#include "postprocess.hpp"
#include <algorithm>
#include <cmath>

namespace rmdetect {

// 解码 YOLO ONNX 输出，提取所有检测结果
// 支持 [1, 4+nc, N] 和 [1, N, 4+nc] 两种常见输出格式
std::vector<Detection> decodeOutput(
    const float        *raw_output,
    const std::vector<int> &output_shape,
    const LetterboxInfo    &lbinfo,
    const Config           &cfg)
{
    std::vector<Detection> detections;

    if (output_shape.size() != 3) {
        return detections; // 不支持的输出格式，返回空
    }

    // 标准格式 [1, 4+num_classes, N]
    int channels  = output_shape[1];
    int num_det   = output_shape[2];
    int exp_channels = 4 + cfg.num_classes;

    if (channels < exp_channels) {
        // 可能是转置格式 [1, N, 4+num_classes]
        if (num_det >= exp_channels && channels <= num_det) {
            int rows = channels;
            int cols = num_det;
            for (int r = 0; r < rows; ++r) {
                const float *row = raw_output + r * cols;
                float cx = row[0], cy = row[1], w = row[2], h = row[3];

                float max_conf = 0.0f;
                int   best_id  = -1;
                for (int c = 0; c < cfg.num_classes; ++c) {
                    float conf = row[4 + c];
                    if (conf > max_conf) { max_conf = conf; best_id = c; }
                }
                if (best_id < 0 || max_conf < cfg.conf_threshold) continue;

                // 坐标从 letterbox 坐标系映射回原图
                float x1 = (cx - w / 2.0f - lbinfo.pad_x) / lbinfo.scale;
                float y1 = (cy - h / 2.0f - lbinfo.pad_y) / lbinfo.scale;
                detections.push_back({best_id, max_conf, x1, y1,
                    w / lbinfo.scale, h / lbinfo.scale,
                    x1 + w / lbinfo.scale / 2.0f,
                    y1 + h / lbinfo.scale / 2.0f});
            }
            return detections;
        }
        return detections;
    }

    // 标准 [1, 4+num_classes, N] 格式
    // 每个候选框存储在列中，stride = num_det
    for (int i = 0; i < num_det; ++i) {
        const float *col = raw_output + i;
        float cx = col[0 * num_det];
        float cy = col[1 * num_det];
        float w  = col[2 * num_det];
        float h  = col[3 * num_det];

        // 找置信度最高的类别
        float max_conf = 0.0f;
        int   best_id  = -1;
        for (int c = 0; c < cfg.num_classes; ++c) {
            float conf = col[(4 + c) * num_det];
            if (conf > max_conf) { max_conf = conf; best_id = c; }
        }

        if (best_id < 0 || max_conf < cfg.conf_threshold) continue;

        // 坐标回映射到原图
        float x1 = (cx - w / 2.0f - lbinfo.pad_x) / lbinfo.scale;
        float y1 = (cy - h / 2.0f - lbinfo.pad_y) / lbinfo.scale;
        float bw = w / lbinfo.scale, bh = h / lbinfo.scale;

        detections.push_back({best_id, max_conf, x1, y1, bw, bh,
                              x1 + bw / 2.0f, y1 + bh / 2.0f});
    }

    return detections;
}

// 非极大值抑制：去除重叠的检测框
std::vector<Detection> nonMaxSuppression(
    std::vector<Detection> &detections, float nms_threshold)
{
    if (detections.empty()) return {};

    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      indices;

    for (const auto &d : detections) {
        boxes.push_back(cv::Rect(
            static_cast<int>(d.x), static_cast<int>(d.y),
            static_cast<int>(d.width), static_cast<int>(d.height)));
        scores.push_back(d.confidence);
    }

    // 调用 OpenCV 自带的 NMS 接口
    cv::dnn::NMSBoxes(boxes, scores, 0.0f, nms_threshold, indices);

    std::vector<Detection> result;
    result.reserve(indices.size());
    for (int idx : indices) {
        result.push_back(detections[idx]);
    }
    return result;
}

} // namespace rmdetect
