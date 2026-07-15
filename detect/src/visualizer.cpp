#include "visualizer.hpp"
#include <opencv2/imgproc.hpp>

namespace rmdetect {

// 在画面中绘制所有检测结果
cv::Mat visualize(const cv::Mat        &frame,
                  const std::vector<Detection> &detections,
                  const Config                 &cfg,
                  double fps)
{
    cv::Mat display = frame.clone(); // 复制一份画布，不修改原图

    // 逐个绘制检测框、中心点和标签
    for (const auto &det : detections) {
        drawOneDetection(display, det, cfg.class_names, det.class_id);
    }

    // 左上角显示当前 FPS
    cv::putText(display,
                cv::format("FPS: %.1f", fps),
                cv::Point(15, 35),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 255, 0), 2);

    // 显示当前帧检测到的目标数量
    cv::putText(display,
                cv::format("目标数: %zu", detections.size()),
                cv::Point(15, 70),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);

    // 没有检测到目标时，画面中央显示红色警告
    if (detections.empty()) {
        cv::putText(display, "未检测到目标",
                    cv::Point(frame.cols / 2 - 80, frame.rows / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2,
                    cv::Scalar(0, 0, 255), 3);
    }

    return display;
}

// 绘制单个检测结果
void drawOneDetection(cv::Mat                    &img,
                      const Detection            &det,
                      const std::vector<std::string> &class_names,
                      int /* class_id */)
{
    cv::Rect box(static_cast<int>(det.x), static_cast<int>(det.y),
                 static_cast<int>(det.width), static_cast<int>(det.height));

    // 根据类别选择颜色：红色装甲板用红色，蓝色装甲板用蓝色
    cv::Scalar color;
    if (det.class_id < static_cast<int>(class_names.size())) {
        const std::string &name = class_names[det.class_id];
        if (name.find("red") != std::string::npos)
            color = cv::Scalar(0, 0, 255);   // 红框
        else if (name.find("blue") != std::string::npos)
            color = cv::Scalar(255, 0, 0);   // 蓝框
        else
            color = cv::Scalar(0, 255, 255);  // 其他类用黄色
    } else {
        color = cv::Scalar(0, 255, 255);
    }

    // 画边界框
    cv::rectangle(img, box, color, 2);

    // 在框上方写标签：类别名 + 置信度
    std::string label = cv::format("%s %.2f",
        (det.class_id < static_cast<int>(class_names.size()))
            ? class_names[det.class_id].c_str() : "未知",
        det.confidence);
    cv::putText(img, label, cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

    // ---- 中心点可视化 ----
    cv::Point center(static_cast<int>(det.cx), static_cast<int>(det.cy));
    cv::circle(img, center, 5, color, -1);                 // 实心圆
    cv::circle(img, center, 5, cv::Scalar(255, 255, 255), 1); // 白色外圈

    // 十字准星
    int cross_len = 10;
    cv::line(img, cv::Point(center.x - cross_len, center.y),
                   cv::Point(center.x + cross_len, center.y), color, 1);
    cv::line(img, cv::Point(center.x, center.y - cross_len),
                   cv::Point(center.x, center.y + cross_len), color, 1);

    // 在中心点旁边标注像素坐标
    cv::putText(img,
                cv::format("(%d, %d)", center.x, center.y),
                cv::Point(center.x + 8, center.y - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
}

} // namespace rmdetect
