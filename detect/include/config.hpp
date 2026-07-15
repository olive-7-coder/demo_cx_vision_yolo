#pragma once

#include <string>
#include <vector>

namespace rmdetect {

// 检测配置结构体 —— 根据你的模型修改这些参数
struct Config {
    std::string model_path = "best.onnx";         // ONNX 模型路径
    int input_width  = 640;                       // 模型输入宽度
    int input_height = 640;                       // 模型输入高度
    int num_classes = 6;                          // 类别总数

    // 类别名列表（必须和阶段一训练时的顺序一致）
    std::vector<std::string> class_names = {
        "blue1", "blue3", "bluesb",
        "red1",  "red3",  "redsb"
    };

    float conf_threshold  = 0.45f;   // 置信度阈值：低于此值的检测结果被过滤
    float nms_threshold   = 0.45f;   // NMS 去重阈值：重叠度大于此值的框合并
    bool use_cuda = false;            // 是否启用 CUDA 加速（需要 OpenCV 带 CUDA）
    bool show     = true;             // 是否弹出 OpenCV 显示窗口
    bool save     = true;             // 是否保存结果到文件
    std::string input_path = "";      // 输入路径：空=摄像头，否则为图片/视频/文件夹路径
    int camera_id = 0;                // 摄像头编号
};

// 单个检测结果的结构体
struct Detection {
    int    class_id;       // 类别编号（0~5）
    float  confidence;     // 置信度（0~1）
    float  x, y;           // 边界框左上角坐标
    float  width, height;  // 边界框的宽和高
    float  cx, cy;         // 中心点坐标
};

} // namespace rmdetect
