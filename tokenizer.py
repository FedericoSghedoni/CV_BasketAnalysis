import math
import torch
import cv2
from ultralytics import YOLO
from pydantic import BaseModel
import utils
import numpy as np

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
        self.rim_coord = torch.zeros([1,4]) # 4 features x,y,w,h
        self.ball_coord = torch.zeros([1,4]) # 4 features x,y,w,h
        self.embedded_feature = torch.Tensor() # rim and ball features (8), joint angle and distance from the rim
        self.detector = YOLO(path_to_model)
        self.pose = YOLO('yolov8n-pose.pt')
        
        self.colors = [(56,56,255), (151,157,255), (31,112,255)] 
        self.labels = ['basketball', 'people', 'rim']              
        self.measured_distance = 1       
        self.real_width = [0.225, 0.49]
        self.real_distance = [0, 0, 0, 0]
        self.pp_data = {}
        self.focal_length = utils.FocalLength(self.measured_distance, self.real_width[0], 'ref.jpg')
        
    def detect_objects(self, frame):
        detections = self.detector.predict(task = 'detect', source = frame, verbose=False, show_conf=False)
        # To show the detection use the line below
        # detections.show()
        # im_array = None
        dist, im_array = self.getDistance(detections, frame)
        # print(f'{dist} dist')
        for detection in detections[0].boxes:
            class_index = int(detection.cls.item())
            if class_index == 0: # Basketball
                self.ball_coord = detection.xywhn
            elif class_index == 2: # Rim
                self.rim_coord = detection.xywhn
        new_feature = torch.cat((self.rim_coord[0], self.ball_coord[0]))

        results = self.pose.predict(frame, save=False, imgsz=640 , conf = 0.5, verbose=False)
        # frame = cv2.cvtColor(im_array, cv2.COLOR_GRAY2BGR)
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

                angle = utils.calculate_angle(shoulder, elbow, wrist)
                if math.isnan(angle):
                    angle = 0
                shooting_angle = torch.tensor([angle])
                # self.embedded_feature = self.embedded_feature.reshape(-1)
                
                new_feature = torch.cat((new_feature, shooting_angle), dim=0)
            else:
                shooting_angle = torch.tensor([0])
                new_feature = torch.cat((new_feature, shooting_angle), dim=0)

        self.embedded_feature = torch.cat((self.embedded_feature, new_feature.unsqueeze(0)), dim=0)
        return im_array, detections[0].boxes
    
    # !! distanza tra rim e player deve essere minore almeno di una tra distanza rim/camera e distanza player/camera, 
    # sempre vero a meno che non si usino grandangoli

    def getDistance(self, results, frame):    
        
        for r in results:
            #im_array = r.plot()  # plot a BGR numpy array of predictions
            count = [0, 0, 0]
            count[0] = r.boxes.cls.tolist().count(0.0)
            count[1] = r.boxes.cls.tolist().count(1.0)
            count[2] = r.boxes.cls.tolist().count(2.0)
        
            for i in range(len(count)-1):
                self.real_distance[2+i] += 1
            for key in self.pp_data.keys():
                self.pp_data[key][2][1] += 1
                self.pp_data[key][3][1] += 1    
                            
            # Itera su ogni riga del tensore
            for row in r.boxes:
                cv2.rectangle(frame, (int(row.data.tolist()[0][0]), int(row.data.tolist()[0][1])), (int(row.data.tolist()[0][2]), int(row.data.tolist()[0][3])), 
                          self.colors[int(row.data.tolist()[0][-1])], 2)
                cv2.putText(frame, self.labels[int(row.data.tolist()[0][-1])], (int(row.data.tolist()[0][0]), int(row.data.tolist()[0][1]) - 7), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[int(row.data.tolist()[0][-1])], 2)
                
                if row.data.tolist()[0][-1] == 0.0:
                    # Crea una lista per la riga corrente e aggiungi i valori
                    riga_box = row.xywh.tolist()[0][:] + [row.data.tolist()[0][-1]]
                    distance = self.focal_length * self.real_width[0] / riga_box[2]
                    self.real_distance[0] = utils.updateDistance(self.real_distance[0], distance, self.real_distance[2])
                    self.real_distance[2] = 0
                    #cv2.putText(im_array, f'C-dist: {self.real_distance[0]:.1f} m', (int(riga_box[0]- (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 30)), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[0], 2)
                    
                elif row.data.tolist()[0][-1] == 1.0:
                    p_h = 0 # person height
                    # Prova a calcolare altezza persona
                    if count[0] >= 1:
                        for bb_box in [box.data.tolist()[0] for box in r.boxes if box.data.tolist()[0][-1] == 0.0]:
                            pp_box = row.data.tolist()[0][:]
                            if utils.check_intersection(bb_box, pp_box):
                                h_pp_img = pp_box[3] - pp_box[1]
                                w_bb_img = bb_box[2] - bb_box[0]
                                bb_dist = self.focal_length * self.real_width[0] / w_bb_img
                                p_h = (h_pp_img * bb_dist) / self.focal_length
                            
                    x,y,w,h = [int(item) for item in (row.xywh.tolist()[0])]
                    roi = [frame[y-int(h*1//2):y+int(h*1//2), x-int(w*1//2):x+int(w*1//2)], [x, y]] # test con *0.9
                    
                    self.pp_data, id = utils.updateData(self.pp_data, roi, p_h)
                    # Crea una lista per la riga corrente e aggiungi i valori
                    riga_box = row.xywh.tolist()[0][:] + [row.data.tolist()[0][-1]]
                    distance = self.focal_length * self.pp_data[id][1][0] / riga_box[3]
                    self.pp_data[id][2][0] = utils.updateDistance(self.pp_data[id][2][0], distance, self.pp_data[id][2][1])
                    # azzero contatore emb
                    self.pp_data[id][2][1] = 0
                    #cv2.putText(im_array, f'{id} C-dist: {self.pp_data[id][2][0]:.1f} m', (int(riga_box[0]- (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 30)), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[1], 2)
                    #cv2.putText(im_array, f'{id} h: {self.pp_data[id][1][0]:.2f} m', (int(riga_box[0]- (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 70)), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[1], 2)
                    
                    if count[2] == 1 and self.pp_data[id][2][0] != 0 and self.real_distance[1] != 0:
                        rim_box = [box.xywh.tolist()[0] for box in r.boxes if box.data.tolist()[0][-1] == 2.0][0] 
                        # Calcola la distanza in pixel tra i due oggetti
                        pixel_dist = abs(riga_box[0] - rim_box[0]) - 10
                        scale_dist = pixel_dist * min(self.pp_data[id][2][0], self.real_distance[1]) / self.focal_length
                        #cv2.line(im_array, (int(riga_box[0]),int(max(riga_box[1], rim_box[1]))), (int(rim_box[0]),int(max(riga_box[1], rim_box[1]))), (0, 0, 255), 2)
                        #p = (self.real_distance[1] + self.real_distance[2] + scale_dist) / 2
                        #h = np.sqrt(p * (p - self.real_distance[1]) * (p - self.real_distance[2]) * (p - scale_dist)) * 2 / max(self.real_distance[1], self.real_distance[2])
                        #b = np.sqrt(1 - (h / min(self.real_distance[1], self.real_distance[2])) ** 2) * min(self.real_distance[1], self.real_distance[2])
                        b = np.sqrt(1 - (scale_dist / min(self.pp_data[id][2][0], self.real_distance[1])) ** 2) * min(self.pp_data[id][2][0], self.real_distance[1])
                        #distance = np.sqrt(h ** 2 + (max(self.real_distance[1], self.real_distance[2]) - b) ** 2)
                        d = np.sqrt(scale_dist ** 2 + (max(self.pp_data[id][2][0], self.real_distance[1]) - b) ** 2)
                        #print(f'{self.real_distance,pixel_dist,scale_dist,p,h,b,distance} self.real_distance, pixel_dist, scale_dist, p, h, b, distance')
                        self.pp_data[id][3][0] = utils.updateDistance(self.pp_data[id][3][0], d, self.pp_data[id][3][1])
                        self.pp_data[id][3][1] = 0
                        # Visualizza la distanza sul frame
                        #cv2.putText(im_array, f'{id} R-dist: {self.pp_data[id][3][0]:.1f} m', (int(riga_box[0]- (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 50)), 
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(frame, f'{id} R-dist: {self.pp_data[id][3][0]:.1f} m', (int(riga_box[0]- (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 24)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[1], 2)
                          
                                
                elif row.data.tolist()[0][-1] == 2.0:
                    # Crea una lista per la riga corrente e aggiungi i valori
                    riga_box = row.xywh.tolist()[0][:] + [row.data.tolist()[0][-1]]
                    distance = self.focal_length * self.real_width[1] / riga_box[2]
                    self.real_distance[1] = utils.updateDistance(self.real_distance[1], distance, self.real_distance[3])
                    self.real_distance[3] = 0
                    #cv2.putText(im_array, f'C-dist: {self.real_distance[1]:.1f} m', (int(riga_box[0] - (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 30)), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[2], 2)

        cv2.destroyAllWindows()
        return self.pp_data, frame

