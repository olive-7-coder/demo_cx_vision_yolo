#pragma once

#include <opencv2/opencv.hpp>

namespace rmdetect {

// letterbox 缩放信息：用于把检测坐标映射回原图
struct LetterboxInfo {
    float scale;      // 缩放比例 = 目标尺寸 / max(原图宽, 原图高)
    int   pad_x;      // 水平方向填充的像素数
    int   pad_y;      // 垂直方向填充的像素数
};

// 对图片做 letterbox 缩放（保持宽高比，不足部分填充灰色）
// 参数 src: 原始图片, blob_out: 输出给模型的 blob
// 参数 info: 返回缩放信息, target_w/h: 模型目标尺寸
// 返回值: letterbox 处理后的图片（用于显示）
cv::Mat letterbox(const cv::Mat &src, cv::Mat &blob_out,
                  LetterboxInfo &info,
                  int target_w, int target_h);

} // namespace rmdetect
