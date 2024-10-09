from ultralytics import YOLO

# Load the YOLOv8 model (pre-trained)
model = YOLO('yolov8n.pt')  # You can use yolov8n (nano), yolov8s (small), etc.

# Train the model
model.train(data='./dataset/data.yaml', epochs=50, imgsz=640)
