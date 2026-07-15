 # RoboMaster 2027 Phase 2 — Armor Plate Detector
 
 ## Overview
 
 This project deploys a YOLO ONNX model trained on RoboMaster armor plates into a C++ inference program. It:
 
 - Loads an ONNX model using OpenCV DNN
 - Detects armor plates (red / blue) in images, videos, or live camera feed
 - Extracts the **center point** of every reliable detection
 - Visualizes bounding boxes, center points (crosshair + dot), class labels, and FPS
 - Saves annotated results to the `results/` folder
 
 This is the **Phase 2** submission for the ZJUT-Deus RoboMaster 2027 vision assessment (大一 freshman track).
 
 ## Directory Structure
 
 ```
 detect/
 ├── CMakeLists.txt         # CMake build configuration
 ├── README.md              # This file
 ├── .gitignore             # Files to exclude from version control
 ├── include/
 │   ├── config.hpp         # Configuration structs and Detection data type
 │   ├── preprocess.hpp     # Image preprocessing (letterbox resize)
 │   ├── detector.hpp       # ONNX model inference wrapper
 │   ├── postprocess.hpp    # YOLO output decoding + NMS
 │   └── visualizer.hpp     # Drawing results on images
 ├── src/
 │   ├── main.cpp           # Entry point (image / video / camera)
 │   ├── preprocess.cpp     # Letterbox implementation
 │   ├── detector.cpp       # ONNX net loading & forward pass
 │   ├── postprocess.cpp    # Output decode & NMS implementation
 │   └── visualizer.cpp     # Drawing implementation
 ├── assets/                # (Optional) test images / short clips
 └── results/               # Saved detection outputs (screenshots, videos)
 ```
 
 ## Requirements
 
 - **C++17** compatible compiler (gcc, clang, MSVC)
 - **CMake >= 3.16**
 - **OpenCV >= 4.5** with `dnn` module enabled
 - Your Phase 1 ONNX model (`best.onnx` or equivalent)
 
 ### Installing OpenCV
 
 **Windows (vcpkg)**
 ```bash
 git clone https://github.com/Microsoft/vcpkg.git
 cd vcpkg && bootstrap-vcpkg.bat
 vcpkg install opencv[core,dnn]:x64-windows
 cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path/to/vcpkg]/scripts/buildsystems/vcpkg.cmake
 ```
 
 **Linux / WSL**
 ```bash
 sudo apt install libopencv-dev
 # or build from source with DNN support
 ```
 
 **macOS**
 ```bash
 brew install opencv
 ```
 
 ## Build
 
 ```bash
 # From the detect/ directory:
 cmake -B build -S .
 cmake --build build --config Release
 ```
 
 The executable `detect` (or `detect.exe` on Windows) will be in `build/Release/` or `build/`.
 
 > **Tip:** Copy your ONNX model into the `detect/` folder (or provide a path at runtime) so the executable can find it.
 
 ## Usage
 
 ### Quick start (camera)
 
 ```bash
 ./detect --model best.onnx
 ```
 
 ### Image input
 
 ```bash
 ./detect --model best.onnx --input test.jpg
 ```
 
 ### Video input
 
 ```bash
 ./detect --model best.onnx --input video.mp4
 ```
 
 ### Camera input
 
 ```bash
 ./detect --model best.onnx --input 0
 ```
 
 ### Batch images in a directory
 
 ```bash
 ./detect --model best.onnx --input ./assets
 ```
 
 ### All options
 
 | Option | Default | Description |
 |---|---|---|
 | `--model <path>` | `model.onnx` | Path to the ONNX model |
 | `--input <path>` | camera 0 | Image file, video file, camera index, or directory |
 | `--conf <float>` | `0.45` | Confidence threshold |
 | `--nms <float>` | `0.45` | NMS IoU threshold |
 | `--classes <int>` | `2` | Number of classes |
 | `--size <w,h>` | `640,640` | Model input size (must match ONNX) |
 | `--names <a,b>` | `red_armor,blue_armor` | Comma-separated class names |
 | `--no-show` | — | Disable the display window |
 | `--no-save` | — | Disable saving results |
 | `--cuda` | — | Use CUDA backend (if available) |
 
 ## Output
 
 - Annotated images saved to `results/result_<input_name>.png`
 - Annotated videos saved to `results/result_<input_name>.mp4`
 - Console shows per-frame detection info (FPS, inference time, center coordinates)
 
 ### Visualization legend
 
 | Element | Color |
 |---|---|
 | Red armor box / crosshair | Red |
 | Blue armor box / crosshair | Blue |
 | FPS / target count overlay | Green |
 | "NO TARGET" warning | Red (center of frame) |
 
 Each detection shows:
 - Bounding box with class label and confidence
 - Filled center dot with white outline
 - Crosshair lines over the center
 - Numeric pixel coordinates `(x, y)` next to the center
 
 ## Model Configuration
 
 Before running, edit the defaults in `include/config.hpp` or pass CLI flags to match your Phase 1 model:
 
 1. **Model path** — where your `.onnx` file lives
 2. **Input size** — e.g., `640×640` (must match export)
 3. **Number of classes** — `2` for red + blue armor, `1` for single, etc.
 4. **Class names** — update to match your training classes (e.g., `red_armor,blue_armor`)
 5. **Confidence / NMS thresholds** — tune based on your model's precision
 
 ## Pipeline Overview
 
 ```
 Input Frame
     │
     ▼
 ┌─────────────────┐
 │  Preprocess      │  Letterbox resize + padding + blob
 │  (letterbox)     │  scale to [0,1], BGR→RGB
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Inference       │  OpenCV DNN forward pass
 │  (ONNX model)    │  output: [1, 4+nc, N] tensor
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Postprocess     │  Decode bbox + confidence + class
 │  (decode + NMS)  │  Map coords back via letterbox info
 └────────┬────────┘          Apply NMS to remove duplicates
          ▼
 ┌─────────────────┐
 │  Center Extract  │  cx = x + w/2, cy = y + h/2
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Visualize       │  Draw boxes, center dots, crosshairs,
 │                  │  class labels, FPS, target count
 └────────┬────────┘
          ▼
     Display / Save
 ```
 
 ## Notes
 
 - The ONNX model file is **not included** in this repo. Place your Phase 1 exported model (`best.onnx`) in this folder or specify its path with `--model`.
 - This project uses **OpenCV DNN** for inference (no external ONNX Runtime dependency). It is the simplest and most portable deployment option recommended by the assignment.
 - No dataset files, build artifacts, or IDE caches are committed (see `.gitignore`).
 - For camera input, press **ESC** or **Q** to exit.

 ## Known Issues
 
 - OpenCV DNN on CPU may be slower than ONNX Runtime or TensorRT. For higher FPS, switch to `--cuda` if your OpenCV was built with CUDA support, or modify `detector.cpp` to use the ONNX Runtime C++ API.
 - Model output shape handling covers common YOLOv8/v10/v11 export formats. If your exported model uses a different output layout, adjust `postprocess.cpp` accordingly.
 - The "NO TARGET" indicator shows when no detections pass the confidence threshold — it does not distinguish between a genuinely empty scene and a missed detection.
