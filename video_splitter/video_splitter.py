import cv2
import numpy as np
import os
from ultralytics import YOLO
import sys
#sys.path.append('C:/Users/sghe9/Desktop/CV_BasketAnalysis')
sys.path.append("C:/Users/alejb/Documents/GitHub/CV_BasketAnalysis")
import utils


# Set working folder path
Path = 'video_splitter/video/'

# Set bin folder path
bin = 'video_splitter/bin/'

# Set path per i files video tagliati
output_path = 'video_splitter/'

# Set model path
model_path = '../yolov8s_final/weights/best.pt'

# Set video name
video_name = 'IMG_4442.MOV'

if not os.path.exists(bin):
    os.makedirs(bin)

# Carica il video
cap = cv2.VideoCapture(f'{Path}{video_name}')

# Carica il modello
model = YOLO(model_path)

save = False
frame_buffer = utils.Buffer(35)
ball_y1 = None  # Coordinata Y della palla
ball_y2 = None  # Coordinata Y della palla
person_y = None  # Coordinata Y della persona
hoop_y = None
ball = None
person = None 
frame_counter = 0
video_counter = 1
video_writer =  None

# Setta writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30

while cap.isOpened():
    # Legge un frame
    ret, frame = cap.read()

    if not ret:
        break
    
    # Ritagliare l'immagine: taglia 190 pixel dall'alto e 370 dal basso
    frame = frame[190:frame.shape[:2][0]-370, :]

    if frame_counter > 90:
        print("FINE 1")
        save = False
        frame_counter = 0

    # Esegue detection
    results = model(frame, conf=0.4, verbose = False)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    for r in results:
        # conta detection
        count = [0, 0, 0]
        count[2] = r.boxes.cls.tolist().count(2.0) #canestro
        count[1] = r.boxes.cls.tolist().count(1.0) #persona
        count[0] = r.boxes.cls.tolist().count(0.0) #palla
            
        if count[2] == 1 and count[1] > 0 and count[0] == 1:
            #print(r.boxes.data.tolist())
            # Itera su ogni riga del tensore
            for row in r.boxes:
                if row.data.tolist()[0][-1] == 0.0:
                    ball_y1 = (row.xyxy[0][1]).numpy()
                    ball_y2 = (row.xyxy[0][3]).numpy()
                    ball = row.xyxy.tolist()[0]
                    #print(row.xyxy[0], "palla")
                elif row.data.tolist()[0][-1] == 1.0:
                    person_y = (row.xyxy[0][1]).numpy()
                    person = row.xyxy.tolist()[0]
                    #print(row.xyxy[0], "persona")   
                elif row.data.tolist()[0][-1] == 2.0:
                    hoop_y = (row.xyxy[0][3]).numpy()
                    #print(row.xyxy[0], "canestro")

            if ball_y2 <= person_y:
                # Inizia a salvare il video
                if not save:
                    print('INIZIO')
                    height, width, channels = frame.shape
                    new_video = 'tiro_{:d}.mp4'.format(video_counter)
                    video_writer = cv2.VideoWriter(output_path + new_video, fourcc, fps, (width, height))
                    save = True
                    video_counter += 1
                    for f in frame_buffer.stack:
                        video_writer.write(f)
                    frame_counter = frame_buffer.size()
                    frame_buffer.clear()

            elif ball_y1 > hoop_y + 5 or ball_y1 > person_y:
                # Interrompi il salvataggio se la condizione non è soddisfatta
                if save and not utils.check_intersection(ball, person):
                    print('FINE 2')
                    save = False
                    video_writer.release()
                    if frame_counter < 45:
                        src = output_path + 'tiro_{:d}.mp4'.format(video_counter-1)
                        dest = bin + 'tiro_{:d}.mp4'.format(video_counter-1)
                        os.rename(src,dest)
                    frame_counter = 0

        if save:
            video_writer.write(frame)
            frame_counter +=1 

        if not save:
            frame_buffer.push(frame)                                
                    
video_writer.release()
cap.release()
cv2.destroyAllWindows()
