import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import utils
import sys

# Set working folder path
Path = 'distance/'

# Set model path
model_path = '../yolov8s_final/weights/best.pt'

# Set video name
video_name = 'IMG_4018.mp4'
size = len(video_name)
new_video = video_name[:size - 4] + 'b.mp4'

# Dimensioni dell'oggetto di riferimento noto (ad es. la palla) in metri  45x50
real_width = [0.218, 0, 0.50, 0]
real_distance = [0, 0, 0, 0, 0, 0]
pr_dist = [0, 0]
colors = [(56,56,255), (151,157,255), (31,112,255)]               
measured_distance = 1
focal_length = utils.FocalLength(measured_distance, real_width[0], 'ref.jpg')

# Carica il video
cap = cv2.VideoCapture(f'{Path}{video_name}')

# Carica il modello
model = YOLO(model_path)

# Legge un frame
ret, frame = cap.read()
# Crea writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
height, width, channels = frame.shape
video_writer = cv2.VideoWriter(f'{Path}{new_video}', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    boxes = []
    # Esegue detection
    results = model(frame, conf=0.4)
    real_width = utils.updateHeight(results, focal_length, real_width)
    #print(f'{real_width} real_width')
    #print(f'{real_distance} real_distance')
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        #print(r.boxes.cls.tolist())
        #print(r)
        count = [0, 0, 0]
        count[0] = r.boxes.cls.tolist().count(0.0)
        count[1] = r.boxes.cls.tolist().count(1.0)
        count[2] = r.boxes.cls.tolist().count(2.0)
        
        for i,c in enumerate(count):
            if c == 0:
                real_distance[3+i] += 1
            
        if count[2] == 1 and count[1] > 0:
            #print(r.boxes.data.tolist())
            # Itera su ogni riga del tensore
            for row in r.boxes:
                #print(row.data.tolist()[0][-1])
                if row.data.tolist()[0][-1] == 0.0:
                    real_distance[3] = 0
                    #print(row.xywh.tolist()[0][:2])
                    # Crea una lista per la riga corrente e aggiungi i valori
                    riga_box = row.xywh.tolist()[0][:]
                    #print(f'{real_width[int(row.data.tolist()[0][-1])]} real width[{int(row.data.tolist()[0][-1])}]')
                    distance = focal_length * real_width[0] / riga_box[2]
                    real_distance = utils.updateCamDist(real_distance, distance, 0)
                    cv2.putText(im_array, f'Dist: {real_distance[0]:.1f} m', (int(riga_box[0]- (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 30)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[0], 2)
                elif row.data.tolist()[0][-1] == 1.0:
                    real_distance[4] = 0
                    # Crea una lista per la riga corrente e aggiungi i valori
                    riga_box = row.xywh.tolist()[0][:]
                    distance = focal_length * real_width[1] / riga_box[3]
                    real_distance = utils.updateCamDist(real_distance, distance, 1)
                    cv2.putText(im_array, f'Dist: {real_distance[1]:.1f} m', (int(riga_box[0]- (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 30)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[1], 2)
                    # Aggiungi la lista alla lista 'boxes'
                    boxes.append(riga_box)
                elif row.data.tolist()[0][-1] == 2.0:
                    real_distance[5] = 0
                    # Crea una lista per la riga corrente e aggiungi i valori
                    riga_box = row.xywh.tolist()[0][:]
                    distance = focal_length * real_width[2] / riga_box[2]
                    real_distance = utils.updateCamDist(real_distance, distance, 2)
                    cv2.putText(im_array, f'Dist: {real_distance[2]:.1f} m', (int(riga_box[0]- (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 30)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[2], 2)
                    # Aggiungi la lista alla lista 'boxes'
                    boxes.append(riga_box)
            #print(boxes)       
            
            # Calcola la distanza in pixel tra i due oggetti
            if real_distance[1] != 0:
                pixel_dist = abs(boxes[0][0] - boxes[1][0])
                scale_dist = pixel_dist * min(real_distance[1], real_distance[2]) / focal_length
                p = (real_distance[1] + real_distance[2] + scale_dist) / 2
                h = np.sqrt(p * (p - real_distance[1]) * (p - real_distance[2]) * (p - scale_dist)) * 2 / max(real_distance[1], real_distance[2])
                #b = np.sqrt(1 - (h / min(real_distance[1], real_distance[2])) ** 2) * min(real_distance[1], real_distance[2])
                b = np.sqrt(1 - (scale_dist / min(real_distance[1], real_distance[2])) ** 2) * min(real_distance[1], real_distance[2])
                #distance = np.sqrt(h ** 2 + (max(real_distance[1], real_distance[2]) - b) ** 2)
                d = np.sqrt(scale_dist ** 2 + (max(real_distance[1], real_distance[2]) - b) ** 2)
                #print(f'{real_distance,pixel_dist,scale_dist,p,h,b,distance} real_distance, pixel_dist, scale_dist, p, h, b, distance')
                pr_dist = utils.updateDistance(pr_dist, d)
                pr_dist[1] = 0
                # Visualizza la distanza sul frame
            cv2.putText(im_array, f'Distanza: {pr_dist[0]:.1f} metri', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        #    cv2.imshow('Immagini', im_array)
        #    k = cv2.waitKey(0)
        #    scelta = k - 48
        #    if scelta == 0:
        #        pass
        #    elif scelta == 1:
        #        pass
        #    elif scelta == -21:
        #        sys.exit()
        
        else:
            pr_dist[1] += 1
            
    video_writer.write(im_array)

video_writer.release()
cap.release()
cv2.destroyAllWindows()

