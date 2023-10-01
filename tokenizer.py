import torch
import math
import cv2
from ultralytics import YOLO
from pydantic import BaseModel
import utils
import numpy as np

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
        self.rim_coord = torch.zeros(4) # 4 features x,y,w,h
        self.ball_coord = torch.zeros(4) # 4 features x,y,w,h
        # self.player_coord = torch.zeros(4) # 4 features x,y,w,h
        self.embedded_feature = torch.Tensor() # rim and ball features (8), joint angle and distance from the rim
        self.detector = YOLO(path_to_model)
        self.pose = YOLO('yolov8n-pose.pt')
        
        self.colors = [(56,56,255), (151,157,255), (31,112,255)]               
        self.measured_distance = 1       
        self.real_width = [0.225, 0, 0.49, 0]
        self.real_distance = [0, 0, 0, 0, 0, 0]
        self.pr_dist = [0, 0]
        self.focal_length = utils.FocalLength(self.measured_distance, self.real_width[0], 'ref.jpg')
        
    def detect_objects(self, frame):
        detections = self.detector(frame) 
        # To show the detection use the line below
        # detections.show()
        dist, im_array = self.getDistance(detections)
        print(f'{dist} dist')
        #frame = cv2.cvtColor(im_array, cv2.COLOR_GRAY2BGR)
        for detection in detections[0].boxes:
            class_index = int(detection.cls.item())
            if class_index == 0: # Basketball
                self.ball_coord = detection.xywhn
            # elif class_index == 1: # People
                # self.player_coord = detection.xywhn
            elif class_index == 2: # Rim
                self.rim_coord = detection.xywhn
        new_feature = torch.cat((self.rim_coord[0], self.ball_coord[0]))

        results = self.pose.predict(frame, save=False, imgsz=640 , conf = 0.5)

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
    
    # !! distanza tra rim e player deve essere minore almeno di una tra distanza rim/camera e distanza player/camera, 
    # sempre vero a meno che non si usino grandangoli

    def getDistance(self, results):    
        boxes = []
        
        self.real_width = utils.updateHeight(results, self.focal_length, self.real_width)
        print(f'{self.real_width[1]} height')
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            count = [0, 0, 0]
            count[0] = r.boxes.cls.tolist().count(0.0)
            count[1] = r.boxes.cls.tolist().count(1.0)
            count[2] = r.boxes.cls.tolist().count(2.0)

            for i,c in enumerate(count):
                if c == 0:
                    self.real_distance[3+i] += 1

            if count[2] == 1 and count[1] > 0:
                # Itera su ogni riga del tensore
                for row in r.boxes:
                    if row.data.tolist()[0][-1] == 0.0:
                        self.real_distance[3] = 0
                        # Crea una lista per la riga corrente e aggiungi i valori
                        riga_box = row.xywh.tolist()[0][:] + [row.data.tolist()[0][-1]]
                        distance = self.focal_length * self.real_width[0] / riga_box[2]
                        self.real_distance = utils.updateCamDist(self.real_distance, distance, 0)
                        cv2.putText(im_array, f'Dist: {self.real_distance[0]:.1f} m', (int(riga_box[0]- (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 30)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[0], 2)
                    elif row.data.tolist()[0][-1] == 1.0:
                        self.real_distance[4] = 0
                        # Crea una lista per la riga corrente e aggiungi i valori
                        riga_box = row.xywh.tolist()[0][:] + [row.data.tolist()[0][-1]]
                        distance = self.focal_length * self.real_width[1] / riga_box[3]
                        self.real_distance = utils.updateCamDist(self.real_distance, distance, 1)
                        cv2.putText(im_array, f'Dist: {self.real_distance[1]:.1f} m', (int(riga_box[0]- (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 30)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[1], 2)
                        # Aggiungi la lista alla lista 'boxes'
                        boxes.append(riga_box)
                    elif row.data.tolist()[0][-1] == 2.0:
                        self.real_distance[5] = 0
                        # Crea una lista per la riga corrente e aggiungi i valori
                        riga_box = row.xywh.tolist()[0][:] + [row.data.tolist()[0][-1]]
                        distance = self.focal_length * self.real_width[2] / riga_box[2]
                        self.real_distance = utils.updateCamDist(self.real_distance, distance, 2)
                        cv2.putText(im_array, f'Dist: {self.real_distance[2]:.1f} m', (int(riga_box[0] - (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 30)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[2], 2)
                        # Aggiungi la lista alla lista 'boxes'
                        boxes.append(riga_box)
                #print(boxes)       

                # Calcola la distanza in pixel tra i due oggetti
                if self.real_distance[1] != 0:
                    pixel_dist = abs(boxes[0][0] - boxes[1][0]) - 10
                    scale_dist = pixel_dist * min(self.real_distance[1], self.real_distance[2]) / self.focal_length
                    cv2.line(im_array, (int(boxes[0][0]),int(max(boxes[0][1], boxes[1][1]))), (int(boxes[1][0]),int(max(boxes[0][1], boxes[1][1]))), (0, 0, 255), 2)
                    b = np.sqrt(1 - (scale_dist / min(self.real_distance[1], self.real_distance[2])) ** 2) * min(self.real_distance[1], self.real_distance[2])
                    d = np.sqrt(scale_dist ** 2 + (max(self.real_distance[1], self.real_distance[2]) - b) ** 2)
                    self.pr_dist = utils.updateDistance(self.pr_dist, d)
                    self.pr_dist[1] = 0
                    # Visualizza la distanza sul frame
                cv2.putText(im_array, f'Distanza: {self.pr_dist[0]:.1f} metri', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            #    cv2.imshow('Immagini', im_array)
            #    k = cv2.waitKey(0)
            #    scelta = k - 48
            #    if scelta == -21:
            #        sys.exit()

            else:
                self.pr_dist[1] += 1

        cv2.destroyAllWindows()
        return self.pr_dist[0], im_array

