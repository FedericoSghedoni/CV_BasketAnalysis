import cv2
import numpy as np
from ultralytics import YOLO
import sys
sys.path.append('C:/Users/sghe9/Desktop/CV_BasketAnalysis')
import utils

# !! distanza tra rim e player deve essere minore almeno di una tra distanza rim/camera e distanza player/camera, 
# sempre vero a meno che non si usino grandangoli

# Set to True to show every frame, False to create a video of the frames with BB
show = True
video_w = False

# Set working folder path
Path = 'distance/'

# Set model path
model_path = '../yolov8s_final/weights/best.pt'

# Set video name
video_name = 'IMG_5000.mp4'
size = len(video_name)
# Nome nuovo video
new_video = video_name[:size - 4] + 'l.mp4'

# Dimensioni dell'oggetto di riferimento noto (ad es. la palla) in metri
real_width = [0.225, 0.49]
# Dizionario che contiene coppie id-[embedding, [h, contatore h per fare media], [dist_cam_p, contatore emb], [dist_r_p, contatore dist]]
pp_data = {}
# Nei primi 3 valori memorizza le distanze delle 3 classi dalla Cam negli ultimi 3 il numero di frame dall'ultimo riconoscimento
real_distance = [0, 0, 0, 0]
# Colori per le BB
colors = [(56,56,255), (151,157,255), (31,112,255)]
# Distanza reale nell'immagine ref.jpg             
measured_distance = 1
# Calcola distanza focale
focal_length = utils.FocalLength(measured_distance, real_width[0], 'ref.jpg')
#
f_n = 0

# Carica il video
cap = cv2.VideoCapture(f'{Path}{video_name}')

# Carica il modello
model = YOLO(model_path)

# Legge un frame
ret, frame = cap.read()

if video_w:
    # Crea writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    height, width, channels = frame.shape
    video_writer = cv2.VideoWriter(f'{Path}{new_video}', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    f_n += 1
    if not ret:
        break
    # Esegue detection
    results = model(frame, conf=0.4, verbose=False)
    
    text = 'frame: ' + str(f_n)
    # Finds space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
    cv2.rectangle(frame, (0,0),  (224+w, 46+h), [0,0,0], -1)
    cv2.putText(frame, text, (8,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        count = [0, 0, 0]
        count[0] = r.boxes.cls.tolist().count(0.0)
        count[1] = r.boxes.cls.tolist().count(1.0)
        count[2] = r.boxes.cls.tolist().count(2.0)
        
        for i in range(len(count)-1):
            real_distance[2+i] += 1
        for key in pp_data.keys():
            pp_data[key][2][1] += 1
            pp_data[key][3][1] += 1    
                        
        # Itera su ogni riga del tensore
        for row in r.boxes:
            if row.data.tolist()[0][-1] == 0.0:
                # Crea una lista per la riga corrente e aggiungi i valori
                riga_box = row.xywh.tolist()[0][:] + [row.data.tolist()[0][-1]]
                distance = focal_length * real_width[0] / riga_box[2]
                real_distance[0] = utils.updateDistance(real_distance[0], distance, real_distance[2])
                real_distance[2] = 0
                
            elif row.data.tolist()[0][-1] == 1.0:
                p_h = 0 # person height
                # Prova a calcolare altezza persona
                if count[0] >= 1:
                    for bb_box in [box.data.tolist()[0] for box in r.boxes if box.data.tolist()[0][-1] == 0.0]:
                        pp_box = row.data.tolist()[0][:]
                        if utils.check_intersection(bb_box, pp_box):
                            h_pp_img = pp_box[3] - pp_box[1]
                            w_bb_img = bb_box[2] - bb_box[0]
                            bb_dist = focal_length * real_width[0] / w_bb_img
                            p_h = (h_pp_img * bb_dist) / focal_length
                        
                x,y,w,h = [int(item) for item in (row.xywh.tolist()[0])]
                roi = [frame[y-int(h*1//2):y+int(h*1//2), x-int(w*1//2):x+int(w*1//2)], [x, y]] # test con *0.9
                
                pp_data, id = utils.updateData(pp_data, roi, p_h)
                # Crea una lista per la riga corrente e aggiungi i valori
                riga_box = row.xywh.tolist()[0][:] + [row.data.tolist()[0][-1]]
                distance = focal_length * pp_data[id][1][0] / riga_box[3]
                pp_data[id][2][0] = utils.updateDistance(pp_data[id][2][0], distance, pp_data[id][2][1])
                # azzero contatore emb
                pp_data[id][2][1] = 0
                
                text = f'id'
                # Finds space required by the text so that we can put a background with that amount of width.
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)   
                cv2.rectangle(frame, (int(row.data.tolist()[0][0] - 2), int(row.data.tolist()[0][1] - 50)), (int(row.data.tolist()[0][0]) + w + 90, int(row.data.tolist()[0][1])+2), 
                                      colors[int(row.data.tolist()[0][-1])], -1)
                cv2.putText(frame, f'{id}', (int(riga_box[0]- (riga_box[2]/2)), int(riga_box[1] - (riga_box[3]/2) - 7)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,255,255), 4)
                cv2.rectangle(frame, (int(row.data.tolist()[0][0]), int(row.data.tolist()[0][1])), (int(row.data.tolist()[0][2]), int(row.data.tolist()[0][3])), 
                          colors[int(row.data.tolist()[0][-1])], 4)
                
                if count[2] == 1 and pp_data[id][2][0] != 0 and real_distance[1] != 0:
                    rim_box = [box.xywh.tolist()[0] for box in r.boxes if box.data.tolist()[0][-1] == 2.0][0] 
                    # Calcola la distanza in pixel tra i due oggetti
                    pixel_dist = abs(riga_box[0] - rim_box[0]) - 10
                    scale_dist = pixel_dist * min(pp_data[id][2][0], real_distance[1]) / focal_length
                    #cv2.line(im_array, (int(riga_box[0]),int(max(riga_box[1], rim_box[1]))), (int(rim_box[0]),int(max(riga_box[1], rim_box[1]))), (0, 0, 255), 2)
                    #p = (real_distance[1] + real_distance[2] + scale_dist) / 2
                    #h = np.sqrt(p * (p - real_distance[1]) * (p - real_distance[2]) * (p - scale_dist)) * 2 / max(real_distance[1], real_distance[2])
                    #b = np.sqrt(1 - (h / min(real_distance[1], real_distance[2])) ** 2) * min(real_distance[1], real_distance[2])
                    b = np.sqrt(1 - (scale_dist / min(pp_data[id][2][0], real_distance[1])) ** 2) * min(pp_data[id][2][0], real_distance[1])
                    #distance = np.sqrt(h ** 2 + (max(real_distance[1], real_distance[2]) - b) ** 2)
                    d = np.sqrt(scale_dist ** 2 + (max(pp_data[id][2][0], real_distance[1]) - b) ** 2)
                    #print(f'{real_distance,pixel_dist,scale_dist,p,h,b,distance} real_distance, pixel_dist, scale_dist, p, h, b, distance')
                    pp_data[id][3][0] = utils.updateDistance(pp_data[id][3][0], d, pp_data[id][3][1])
                    pp_data[id][3][1] = 0
                    
                #print(f'{pp_data, id} pp_data, id')  
                               
            elif row.data.tolist()[0][-1] == 2.0:
                # Crea una lista per la riga corrente e aggiungi i valori
                riga_box = row.xywh.tolist()[0][:] + [row.data.tolist()[0][-1]]
                distance = focal_length * real_width[1] / riga_box[2]
                real_distance[1] = utils.updateDistance(real_distance[1], distance, real_distance[3])
                real_distance[3] = 0

        if show: 
            cv2.namedWindow('Immagini', cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('Immagini', 540, 960)
            cv2.imshow('Immagini', frame)
            k = cv2.waitKey(0)
            scelta = k - 48
            if scelta == 0:
                pass
            elif scelta == 1:
                cv2.imwrite(f'distance/frame{f_n}.jpg', frame)
            elif scelta == -21:
                sys.exit()
            
    if video_w:        
        video_writer.write(im_array)

if video_w:        
    video_writer.release()
cap.release()
cv2.destroyAllWindows()

