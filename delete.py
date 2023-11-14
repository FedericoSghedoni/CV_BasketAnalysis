import json 
from tokenizer import Tokenizer
import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
import sys

# Get the video files in the folder
folder_path= 'dataset/canestro/'
classes = torch.Tensor([1., 2., 3.])
folder = 'json/'
dest_folder = folder + folder_path.split('/')[1] + '/'

def report(file,data):
    file = file + '.json'
    print(data)
    # Scrivere l'oggetto JSON nel file
    with open(dest_folder+file, 'a') as json_file:
        json.dump(data, json_file, indent=2)  # L'argomento 'indent' aggiunge la formattazione per una migliore leggibilità

tok = Tokenizer(path_to_model='yolov8s_final/weights/best.pt')


# Crea le cartelle di destinazione se non esistono già
for f in ['', 'canestro/', 'fuori/']:
    if not os.path.exists(os.path.join(folder, f)):
        os.makedirs(os.path.join(folder, f))


for video_file in [f for f in os.listdir(folder_path) if f.endswith('.mp4')]:
    video_path = os.path.join(folder_path, video_file)
    size = len(video_file)
    name = video_file[:size - 4]
    file = name + '.json'
    # Scrivere l'oggetto JSON nel file
    with open(dest_folder+file, 'w') as json_file:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            continue

        # Read and process frames one at a time
        i = 0 # number of frames seen, used for naming the new files
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of the video file
            
            i += 1 # add 1 to the count

            __ = tok.detect_objects(frame)
            
            # Converti l'oggetto Tensor in una lista o NumPy array
            data = tok.embedded_feature.tolist()
            tok.embedded_feature = torch.Tensor()
            # Ora puoi serializzare la lista o l'array usando json.dumps
            json.dump({i: data}, json_file)

            # Ora puoi scrivere il JSON in un file o fare altro con esso
            #json_file.write(",")  # Aggiungi una virgola se non è l'ultima riga
            #json_file.write('\n')  # Aggiungi un carattere di nuova linea tra le righe
            print(i)
    cap.release()
