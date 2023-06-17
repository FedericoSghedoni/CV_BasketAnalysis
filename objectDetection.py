from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='dataset/data.yaml', epochs=100, imgsz=416, batch=16, name='yolov8s_custom')
model.val()
model.export(format="onnx")