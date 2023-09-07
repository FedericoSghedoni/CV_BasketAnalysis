import torch
import math

from ultralytics import YOLO
from pydantic import BaseModel

def calculate_angle(point1, point2, point3):
    # Calcola il vettore tra point2 e point1
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    norm1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    norm2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    angle_radians = math.acos(dot_product / (norm1 * norm2))
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees / 180 # Normaalized Value

class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

class Tokenizer():

    def __init__(self, path_to_model) -> None:
        self.frame_numb = 0
        self.rim_coord = torch.zeros(4) # 4 features x,y,w,h
        self.ball_coord = torch.zeros(4) # 4 features x,y,w,h
        # self.player_coord = torch.zeros(4) # 4 features x,y,w,h
        self.embedded_feature = torch.Tensor() # rim and ball features (8), joint angle and distance from the rim
        self.detector = YOLO(path_to_model)
        self.pose = YOLO('yolov8n-pose.pt')
        
    def detect_objects(self, frame):
        detections = self.detector(frame)
        self.frame_numb +=1 
        # To show the detection use the line below
        # detections.show()
        for detection in detections[0].boxes:
            class_index = int(detection.cls.item())
            if class_index == 0: # Basketball
                self.ball_coord = detection.xywhn
            # elif class_index == 1: # People
                # self.player_coord = detection.xywhn
            elif class_index == 2: # Rim
                self.rim_coord = detection.xywhn
        new_feature = torch.cat([torch.Tensor([self.frame_numb]), self.rim_coord[0], self.ball_coord[0]])

        results = self.pose.predict(frame, save=False, imgsz=320 , conf = 0.5)

        for r in results:

            result_keypoint = r.keypoints.xyn.cpu().numpy()[0]

            if len(result_keypoint) != 0:
            
                get_keypoint = GetKeypoint()
                x_elbow, y_elbow = result_keypoint[get_keypoint.RIGHT_ELBOW]
                x_wrist, y_wrist = result_keypoint[get_keypoint.RIGHT_WRIST]
                x_shoulder, y_shoulder = result_keypoint[get_keypoint.RIGHT_SHOULDER]

                elbow = (x_elbow, y_elbow)  # Sostituisci con le coordinate del gomito
                wrist = (x_wrist, y_wrist)  # Sostituisci con le coordinate del polso
                shoulder = (x_shoulder, y_shoulder)  # Sostituisci con le coordinate della spalla

                angle = calculate_angle(shoulder, elbow, wrist)
                shooting_angle = torch.tensor([angle])
                self.embedded_feature = self.embedded_feature.reshape(-1)
                
                new_feature = torch.cat((new_feature, shooting_angle), dim=0)
        self.embedded_feature = torch.cat((self.embedded_feature, new_feature))
        return detections[0].plot(), detections[0].boxes