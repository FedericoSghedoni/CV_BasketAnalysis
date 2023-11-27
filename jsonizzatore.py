import json 
from tokenizer import Tokenizer
import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
import sys

# Get the video files in the folder
folder_path= '../CVDataset/transformer_dataset/fuori/'
classes = torch.Tensor([1., 2., 3.])
folder = 'json/'
dest_folder = folder + folder_path.split('/')[3] + '/'
max_length, video_name = (160,'')

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
    show = False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        continue
    dati = {}
    
    # Read and process frames one at a time
    while True:
        ret, frame = cap.read()
        
        # here the magic happens
        if not ret:
            # j number of frames seen, used for naming the new files + the padding up to 160
            for j in range(160):
                
                if j > max_length:
                    max_length, video_name = j, video_path

                index = 'frame_' + str(j)
                if j < tok.embedded_feature.shape[0]:
                    dati[index] = tok.embedded_feature[j].numpy().tolist()
                else: dati[index] = [0,0,0,0,0,0,0,0,0]
            with open(dest_folder + file, 'w') as wfile:
                json.dump(dati, wfile)
                print(f'Data written for {file}')

                # reset the embedded features for the next video
                tok.embedded_feature = torch.Tensor()
            break  # End of the video file
        if frame.shape[:2][0] != frame.shape[:2][1]:
            # Ritagliare l'immagine: taglia 190 pixel dall'alto e 370 dal basso
            frame = frame[190:frame.shape[:2][0]-370, :]
        if show:
            cv2.imshow('belllaaaa',frame)
            cv2.waitKey(0)
            show=False
        # pass all the frames to the tokenizer to get the embedded features
        tok.detect_objects(frame)

    cap.release()
print(f'The maximum length readed is {max_length} in the {video_path} file')