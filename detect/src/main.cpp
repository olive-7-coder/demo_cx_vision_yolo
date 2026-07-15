#include <iostream>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "detector.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"
#include "visualizer.hpp"

namespace fs = std::filesystem;

// 打印帮助信息
static void printUsage(const char *prog) {
    std::cout << "\nRoboMaster 装甲板检测器 — 阶段二\n"
              << "用法:\n"
              << "  " << prog << " [选项]\n\n"
              << "选项:\n"
              << "  --model <路径>   ONNX 模型路径 (默认: model.onnx)\n"
              << "  --input <路径>   输入: 图片文件、视频文件、摄像头编号 (默认: 摄像头 0)\n"
              << "  --conf <数值>    置信度阈值 (默认: 0.45)\n"
              << "  --nms  <数值>    NMS 阈值 (默认: 0.45)\n"
              << "  --classes <数值> 类别数 (默认: 6)\n"
              << "  --size <宽,高>   模型输入尺寸 (默认: 640,640)\n"
              << "  --names <c1,c2>  类别名 (默认: red_armor,blue_armor)\n"
              << "  --no-show        不弹窗显示\n"
              << "  --no-save        不保存结果\n"
              << "  --cuda           启用 CUDA 加速\n"
              << "  --help           显示帮助\n\n"
              << "示例:\n"
              << "  " << prog << " --model best.onnx --input test.jpg\n"
              << "  " << prog << " --model best.onnx --input video.mp4\n"
              << "  " << prog << " --model best.onnx --input 0\n"
              << std::endl;
}

// 解析命令行参数
static bool parseArgs(int argc, char **argv, rmdetect::Config &cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            printUsage(argv[0]);
            return false;
        }

        auto next = [&]() -> std::string {
            if (++i >= argc) {
                std::cerr << "缺少参数值: " << arg << std::endl;
                exit(1);
            }
            return argv[i];
        };

        if (arg == "--model")   cfg.model_path = next();
        else if (arg == "--input")   cfg.input_path = next();
        else if (arg == "--conf")    cfg.conf_threshold = std::stof(next());
        else if (arg == "--nms")     cfg.nms_threshold  = std::stof(next());
        else if (arg == "--classes") cfg.num_classes    = std::stoi(next());
        else if (arg == "--size") {
            std::string s = next();
            auto comma = s.find(',');
            cfg.input_width  = std::stoi(s.substr(0, comma));
            cfg.input_height = std::stoi(s.substr(comma + 1));
        }
        else if (arg == "--names") {
            std::string s = next();
            cfg.class_names.clear();
            size_t pos = 0;
            while ((pos = s.find(',')) != std::string::npos) {
                cfg.class_names.push_back(s.substr(0, pos));
                s.erase(0, pos + 1);
            }
            if (!s.empty()) cfg.class_names.push_back(s);
        }
        else if (arg == "--no-show")  cfg.show = false;
        else if (arg == "--no-save")  cfg.save = false;
        else if (arg == "--cuda")     cfg.use_cuda = true;
        else {
            std::cerr << "未知选项: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

// 处理单张图片
static void processImage(const std::string &path,
                         rmdetect::Detector &detector,
                         rmdetect::Config   &cfg) {
    cv::Mat frame = cv::imread(path);
    if (frame.empty()) {
        std::cerr << "[主程序] 无法加载图片: " << path << std::endl;
        return;
    }

    // 1. 预处理
    rmdetect::LetterboxInfo lbinfo;
    cv::Mat blob;
    rmdetect::letterbox(frame, blob, lbinfo,
                        detector.inputWidth(), detector.inputHeight());

    // 2. 推理 + 计时
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> out_shape;
    std::vector<float> raw_out = detector.infer(blob, out_shape);
    auto t1 = std::chrono::high_resolution_clock::now();
    float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // 3. 后处理（解码 + NMS）
    auto dets = rmdetect::decodeOutput(raw_out.data(), out_shape, lbinfo, cfg);
    dets = rmdetect::nonMaxSuppression(dets, cfg.nms_threshold);

    // 打印检测结果
    std::cout << "\n── 图片: " << path << " ──" << std::endl;
    std::cout << "推理耗时: " << infer_ms << " ms" << std::endl;
    std::cout << "检测数量: " << dets.size() << std::endl;
    for (const auto &d : dets) {
        std::cout << "  类别=" << d.class_id
                  << " 置信度=" << d.confidence
                  << " 中心点=(" << d.cx << ", " << d.cy << ")"
                  << std::endl;
    }

    // 4. 可视化（单张图没有 FPS，按 1000/推理耗时 模拟）
    double fps = 1000.0 / std::max(infer_ms, 0.001f);
    cv::Mat result = rmdetect::visualize(frame, dets, cfg, fps);

    // 5. 保存结果
    if (cfg.save) {
        fs::path in(path);
        std::string out_name = "result_" + in.stem().string() + ".png";
        fs::path out_path = fs::path("results") / out_name;
        cv::imwrite(out_path.string(), result);
        std::cout << "已保存: " << out_path.string() << std::endl;
    }

    // 6. 显示结果（按任意键继续）
    if (cfg.show) {
        cv::imshow("RoboMaster 检测器", result);
        std::cout << "按任意键继续..." << std::endl;
        cv::waitKey(0);
    }
}

// 处理视频文件或摄像头流
static void processVideo(const std::string &path,
                         rmdetect::Detector &detector,
                         rmdetect::Config   &cfg) {
    cv::VideoCapture cap;
    bool is_camera = false;

    // 判断输入是否为摄像头编号（纯数字字符串）
    if (!path.empty()) {
        int cam_idx = -1;
        try {
            size_t pos;
            cam_idx = std::stoi(path, &pos);
            if (pos == path.length()) is_camera = true;
        } catch (...) {}
    }

    if (is_camera) {
        int idx = std::stoi(path);
        cap.open(idx);
        if (!cap.isOpened()) {
            std::cerr << "[主程序] 无法打开摄像头 " << idx << std::endl;
            return;
        }
        std::cout << "[主程序] 摄像头 " << idx << " 已打开" << std::endl;
    } else {
        cap.open(path);
        if (!cap.isOpened()) {
            std::cerr << "[主程序] 无法打开视频: " << path << std::endl;
            return;
        }
        std::cout << "[主程序] 视频已打开: " << path << std::endl;
    }

    // 准备视频写入器（保存结果视频）
    cv::VideoWriter writer;
    std::string out_video_path;
    if (cfg.save) {
        fs::path in(path);
        out_video_path = "results/result_" + in.stem().string() + ".mp4";
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double in_fps = cap.get(cv::CAP_PROP_FPS);
        if (in_fps <= 0) in_fps = 30.0;
        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        writer.open(out_video_path, fourcc, in_fps, cv::Size(w, h));
        if (!writer.isOpened()) {
            std::cerr << "[主程序] 无法创建输出视频" << std::endl;
        }
    }

    // 计时相关
    using Clock = std::chrono::high_resolution_clock;
    auto t_last = Clock::now();
    int frame_count = 0;
    double total_infer_ms = 0.0;

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        // 1. 预处理
        rmdetect::LetterboxInfo lbinfo;
        cv::Mat blob;
        rmdetect::letterbox(frame, blob, lbinfo,
                            detector.inputWidth(), detector.inputHeight());

        // 2. 推理 + 计时
        auto t0 = Clock::now();
        std::vector<int> out_shape;
        std::vector<float> raw_out = detector.infer(blob, out_shape);
        auto t1 = Clock::now();
        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        total_infer_ms += infer_ms;
        frame_count++;

        // 3. 后处理
        auto dets = rmdetect::decodeOutput(raw_out.data(), out_shape, lbinfo, cfg);
        dets = rmdetect::nonMaxSuppression(dets, cfg.nms_threshold);

        // 计算当前 FPS
        auto t_now = Clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_last).count();
        double fps = (elapsed > 0) ? 1.0 / elapsed : 0.0;
        t_last = t_now;

        // 4. 可视化
        cv::Mat result = rmdetect::visualize(frame, dets, cfg, fps);

        // 5. 显示
        if (cfg.show) {
            cv::imshow("RoboMaster 检测器", result);
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') break; // ESC 或 Q 退出
        }

        // 6. 保存帧
        if (writer.isOpened()) {
            writer.write(result);
        }

        // 每 30 帧在控制台输出一次统计信息
        if (frame_count % 30 == 0) {
            float avg_infer = total_infer_ms / frame_count;
            std::cout << "帧 " << frame_count
                      << " | FPS: " << fps
                      << " | 平均推理: " << avg_infer << " ms"
                      << " | 目标数: " << dets.size()
                      << std::endl;
        }
    }

    cap.release();
    if (writer.isOpened()) writer.release();

    float avg_infer = (frame_count > 0) ? (total_infer_ms / frame_count) : 0.0f;
    std::cout << "\n── 汇总 ──" << std::endl;
    std::cout << "处理帧数: " << frame_count << std::endl;
    std::cout << "平均推理耗时: " << avg_infer << " ms" << std::endl;
    if (cfg.save) {
        std::cout << "结果已保存: " << out_video_path << std::endl;
    }
}

// 主函数
int main(int argc, char **argv) {
    rmdetect::Config cfg;

    if (!parseArgs(argc, argv, cfg)) {
        return (argc > 1) ? 1 : 0;
    }

    // 检查模型文件是否存在
    if (!fs::exists(cfg.model_path)) {
        std::cerr << "[主程序] 模型文件不存在: " << cfg.model_path << std::endl;
        return 1;
    }

    // 检查类别名数量是否和类别数一致
    if (static_cast<int>(cfg.class_names.size()) != cfg.num_classes) {
        std::cout << "[主程序] 警告: 类别名数量 (" << cfg.class_names.size()
                  << ") != 类别数 (" << cfg.num_classes << ")" << std::endl;
        cfg.class_names.resize(cfg.num_classes, "unknown");
    }

    // 创建结果输出目录
    fs::create_directories("results");

    // 打印配置信息
    std::cout << "\n=== RoboMaster 装甲板检测器 ===\n"
              << "模型: "   << cfg.model_path << "\n"
              << "输入尺寸: " << cfg.input_width << "x" << cfg.input_height << "\n"
              << "类别数: " << cfg.num_classes << "\n"
              << "置信度阈值: " << cfg.conf_threshold << "\n"
              << "NMS阈值: "  << cfg.nms_threshold << "\n"
              << std::endl;

    // 初始化模型
    rmdetect::Detector detector(cfg);

    // 判断输入源并执行
    if (cfg.input_path.empty()) {
        // 未指定输入 → 默认尝试摄像头 0
        std::cout << "[主程序] 未指定输入，尝试打开摄像头 0..." << std::endl;
        processVideo("0", detector, cfg);
    } else {
        // 检查是否为文件夹
        if (fs::is_directory(cfg.input_path)) {
            std::cout << "[主程序] 处理文件夹: " << cfg.input_path << std::endl;
            for (const auto &entry : fs::directory_iterator(cfg.input_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                        processImage(entry.path().string(), detector, cfg);
                    }
                }
            }
        }
        // 检查是否为图片文件
        else {
            std::string ext = fs::path(cfg.input_path).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                processImage(cfg.input_path, detector, cfg);
            }
            // 否则当作视频或摄像头处理
            else {
                processVideo(cfg.input_path, detector, cfg);
            }
        }
    }

    cv::destroyAllWindows();
    std::cout << "\n处理完成。结果已保存到 results/" << std::endl;
    return 0;
}
