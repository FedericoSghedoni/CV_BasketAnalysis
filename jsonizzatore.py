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
            # j number of frames seen, used for naming the new files
            for j in range(tok.embedded_feature.shape[0]):
                
                if j >= 110: print(j)

                index = 'frame_' + str(j)
                dati[index] = tok.embedded_feature[j].numpy().tolist()
            with open(dest_folder + file, 'w') as wfile:
                json.dump(dati, wfile)
                print(f'Data written for {file}')

                # reset the embedded features for the next video
                tok.embedded_feature = torch.Tensor()
            break  # End of the video file
        
        # pass all the frames to the tokenizer to get the embedded features
        tok.detect_objects(frame)

    cap.release()