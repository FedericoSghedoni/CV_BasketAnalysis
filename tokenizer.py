import torch
from ultralytics import YOLO

class Tokenizer():

    def __init__(self, path_to_model) -> None:
        self.embedded_feature = torch.zeros(4)
        self.model = YOLO(path_to_model)
        
    def detect_objects(self, frame):
        detections = self.model(frame)
        # kalman_filter = Kalman(height=frame.shape[0],width=frame.shape[1])
        # To show the detection use the line below
        # detections.show()
        self.embedded_feature = detections[0].boxes.xywhn
        print(self.embedded_feature)
        return detections[0].plot(), detections[0].boxes