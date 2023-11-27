import cv2
import numpy as np
from ultralytics import YOLO
import sys
sys.path.append("C:/Users/alejb/Documents/GitHub/CV_BasketAnalysis")
import utils


class Buffer:
    def __init__(self, max_length):
        self.stack = []
        self.max_length = max_length

    def push(self, item):
        self.stack.append(item)
        if len(self.stack) > self.max_length:
            self.stack.pop(0)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            return None

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)
    
    def clear(self):
        self.stack = []

# !! distanza tra rim e player deve essere minore almeno di una tra distanza rim/camera e distanza player/camera, 
# sempre vero a meno che non si usino grandangoli

# Set working folder path
Path = 'C:/Users/alejb/Documents/GitHub/CV_BasketAnalysis/video_splitter/video/'

# Set model path
model_path = '../yolov8s_final/weights/best.pt'

# Set video name
video_name = 'IMG_4442.MOV'
size = len(video_name)
new_video = video_name[:size - 4] + 'l.mp4'

# Carica il video
cap = cv2.VideoCapture(f'{Path}{video_name}')

# Carica il modello
model = YOLO(model_path)

# Legge un frame
ret, frame = cap.read()

# Imposta il percorso completo per il file video tagliato
output_path = 'C:/Users/alejb/Documents/GitHub/CV_BasketAnalysis/video_splitter/'

video_counter = 1

# Crea writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
height, width, channels = frame.shape
video_name = 'tiro_{:d}.mp4'.format(video_counter)
video_writer = cv2.VideoWriter(output_path + video_name,  fourcc, fps, (width, height))


save = False
frame_buffer = Buffer(35)
ball_y1 = None  # Coordinata Y della palla
ball_y2 = None  # Coordinata Y della palla
person_y = None  # Coordinata Y della persona
hoop_y = None
ball = None
person = None 
frame_counter = 0


while cap.isOpened():
    ret, frame = cap.read()

    # Ritagliare l'immagine: taglia 190 pixel dall'alto e 370 dal basso
    frame = frame[190:frame.shape[:2][0]-370, :]
    if not ret:
        break

    if frame_counter > 45:
        #print("FINE")
        save = False
        frame_counter = 0

    # Esegue detection
    results = model(frame, conf=0.4, verbose = False)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    #cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        #print(r.boxes.cls.tolist())
        #print(r)
        #ordine detection
        count = [0, 0, 0]
        count[2] = r.boxes.cls.tolist().count(2.0) #canestro
        count[1] = r.boxes.cls.tolist().count(1.0) #persona
        count[0] = r.boxes.cls.tolist().count(0.0) #palla
            
        if count[2] == 1 and count[1] > 0 and count[0] == 1:
            #print(r.boxes.data.tolist())
            # Itera su ogni riga del tensore
            for row in r.boxes:
               
                if row.data.tolist()[0][-1] == 0.0:
                    #real_distance[3] = 0
                    #print(row.xywh.tolist()[0][:2])
                    # Crea una lista per la riga corrente e aggiungi i valori
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
                    #print('INIZIO')
                    video_name = 'tiro_{:d}.mp4'.format(video_counter)
                    video_writer = cv2.VideoWriter(output_path + video_name, fourcc, fps, (width, height))
                    save = True
                    video_counter += 1
                    for f in frame_buffer.stack:
                        video_writer.write(f)
                    frame_buffer.clear()

                
            elif ball_y1 > hoop_y + 5 or ball_y1 > person_y:
                # Interrompi il salvataggio se la condizione non è soddisfatta
                #print(f'{utils.check_intersection(ball, person), dir} check, dir')
                if save and not utils.check_intersection(ball, person):
                    #print('FINE')
                    save = False
                    frame_counter = 0

        if save:
            video_writer.write(frame)
            frame_counter +=1 

        if not save:
            frame_buffer.push(frame)                                
                    
video_writer.release()
cap.release()
cv2.destroyAllWindows()

