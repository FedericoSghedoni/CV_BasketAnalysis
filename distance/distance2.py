import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import sys

Path = ''
model_path = '../yolov8s_final/weights/best.pt'
video_name = 'IMG_4019.mp4'
size = len(video_name)
new_video = video_name[:size - 4] + 'b.mp4'
# Dimensioni dell'oggetto di riferimento noto (ad es. una scheda) in millimetri
oggetto_di_riferimento_larghezza_mm = 450
oggetto_di_riferimento_altezza_mm = 500

# Carica i file di calibrazione
camera_matrix = np.load('../calibration/cameraMatrix.pkl', allow_pickle=True)
dist_coeffs = np.load('../calibration/dist.pkl', allow_pickle=True)

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
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        #print(r.boxes.cls.tolist())
        count_rim = r.boxes.cls.tolist().count(2.0)
        count_people = r.boxes.cls.tolist().count(1.0)
        if count_rim == 1 and count_people > 0:
            #print(r.boxes.data.tolist())
            # Itera su ogni riga del tensore
            for row in r.boxes:
                #print(row.data.tolist()[0][-1])
                if row.data.tolist()[0][-1] != 0.0:
                    #print(row.xywh.tolist()[0][:2])
                    # Crea una lista per la riga corrente e aggiungi i valori
                    riga_box = row.xywh.tolist()[0][:2]
                    # Aggiungi la lista alla lista 'boxes'
                    boxes.append(riga_box)
            #print(boxes)       

            # Calcola la distanza tra due oggetti (ad esempio, oggetto 0 e oggetto 1) 
            #object_0 = boxes[0]
            #object_1 = boxes[1]
            x0, y0 = boxes[0][:]
            x1, y1 = boxes[1][:]
            # Applica la trasformazione delle coordinate del pixel alle coordinate del mondo reale
            object_0_pixel = np.array([[x0, y0]], dtype='float32')
            object_1_pixel = np.array([[x1, y1]], dtype='float32')
            object_0_real = cv2.undistortPoints(object_0_pixel, camera_matrix, dist_coeffs)
            object_1_real = cv2.undistortPoints(object_1_pixel, camera_matrix, dist_coeffs)
            # Calcola la distanza euclidea tra i due oggetti
            distance = np.linalg.norm(object_1_real - object_0_real) * 8  #!!!!!!!!
            # Visualizza la distanza sul frame
            cv2.putText(im_array, f'Distanza: {distance:.2f} metri', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            #cv2.imshow('Immagini', im_array)
            #k = cv2.waitKey(0)
            #scelta = k - 48
            #if scelta == 0:
            #    pass
            #elif scelta == 1:
            #    pass
            #elif scelta == -21:
            #    sys.exit()
 
    video_writer.write(im_array)





video_writer.release()
cap.release()
cv2.destroyAllWindows()
