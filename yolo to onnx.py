from ultralytics import YOLO
model = YOLO(r"runs\detect\train-3\weights\best.pt")
model.export(format='onnx')