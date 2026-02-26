from ultralytics import YOLO

# Start with standard model for stability
model = YOLO("yolov8n.pt")

# Export to TensorRT engine (FP16 optimized for RTX)
model.export(format="engine", half=True, imgsz=640)

print("TensorRT export complete.")