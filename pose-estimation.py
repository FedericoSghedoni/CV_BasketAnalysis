from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

source = "C:/Users/acer/Documents/GitHub/CV_BasketAnalysis/dataset/ours/video/video1.mp4"

results = model.predict(source, save=True, imgsz=320 , conf = 0.5)

for r in results:
    print(r.keypoints.cpu().numpy()[0]) 