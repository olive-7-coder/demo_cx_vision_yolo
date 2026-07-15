#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "config.hpp"

namespace rmdetect {

// 在原图画面上绘制检测结果
// 参数 frame: 原始图片
// 参数 detections: 检测结果列表
// 参数 cfg: 配置（类别名等）
// 参数 fps: 当前帧率
// 返回值: 绘制后的图片
cv::Mat visualize(const cv::Mat        &frame,
                  const std::vector<Detection> &detections,
                  const Config                 &cfg,
                  double fps);

// 绘制单个检测结果：边界框、类别标签、置信度、中心点
// 中心点包含：实心圆 + 白色描边 + 十字准星 + 坐标文字
void drawOneDetection(cv::Mat                    &img,
                      const Detection            &det,
                      const std::vector<std::string> &class_names,
                      int class_id);

} // namespace rmdetect
