#include "preprocess.hpp"

namespace rmdetect {

cv::Mat letterbox(const cv::Mat &src, cv::Mat &blob_out,
                  LetterboxInfo &info,
                  int target_w, int target_h) {
    int src_w = src.cols;  // 原图宽度
    int src_h = src.rows;  // 原图高度

    // 计算缩放比例：把较长的边缩放到目标尺寸
    float scale = std::min(static_cast<float>(target_w) / src_w,
                           static_cast<float>(target_h) / src_h);
    int   new_w = static_cast<int>(src_w * scale);  // 缩放后的宽度
    int   new_h = static_cast<int>(src_h * scale);  // 缩放后的高度

    // 等比缩放图片
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 计算需要填充的像素数（让图片变成正方形）
    int pad_w = target_w - new_w;
    int pad_h = target_h - new_h;
    int pad_left = pad_w / 2;
    int pad_top  = pad_h / 2;

    // 保存缩放信息，供后续把检测坐标映射回原图
    info.scale = scale;
    info.pad_x = pad_left;
    info.pad_y = pad_top;

    // 在缩放后的图片四周填充灰色边框
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_top, pad_h - pad_top,
                       pad_left, pad_w - pad_left,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 转换为模型输入需要的 blob 格式
    // 1 × 3 × H × W，BGR → RGB，像素值缩放到 [0,1]
    cv::dnn::blobFromImage(padded, blob_out, 1.0 / 255.0,
                           cv::Size(target_w, target_h),
                           cv::Scalar(), true, false);

    return padded; // 返回 letterbox 后的图片，用于显示
}

} // namespace rmdetect
