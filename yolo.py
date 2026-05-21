from ultralytics import YOLO
# 加载预训练 YOLO 模型 (有不同版本和⼤⼩可选)
model = YOLO("yolo11n.pt")
# 在你⾃⼰的数据集上训练
model.train(data="./My First Project.v1-1.yolov8/data.yaml", epochs=200, imgsz=640)