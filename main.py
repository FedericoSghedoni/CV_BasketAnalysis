import cv2
import numpy as np
import torch

from kalmanFilter import Kalman
from transformer import Transformer
from tokenizer import Tokenizer

# Main Loop

model_path = 'yolov8s_final/weights/best.pt'
model_directory = 'result/model.pt'
# input tensor dimension for the transformer
input_dimension = 160

cap = cv2.VideoCapture('../CVDataset/transformer_dataset/canestro/3.mp4')
ret, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24
frame_width = frame.shape[1]
frame_height = frame.shape[0]
video_writer = cv2.VideoWriter('video_detections2.mp4', fourcc, fps, (frame_width, frame_height))
i = 0

tokenizer = Tokenizer(model_path)
transformer = Transformer(tgt_size=1, n_feature=9,  d_model=160)
kalman_filter = Kalman(height=frame_height, width=frame_width)

while cap.isOpened():
    i += 1
    ret, frame = cap.read()
    if not ret: # or i == 10:
        break

    # Rileva e traccia gli oggetti nel frame
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    new_frame, detection = tokenizer.detect_objects(frame=frame)
    for result in detection:
        # Get the object class
        class_index = int(result.cls.item())

        # If the object is a Basketball make a prediction of its trajectory
        if class_index == 0:
            # Get the bounding box coordinates
            x,y = result.xywh[0,0:2]
            kalman_filter.makePrediction(x,y)
    video_writer.write(new_frame)

pad_width = ((0, input_dimension - tokenizer.embedded_feature.shape[0]), (0, 0))
input_tens = np.pad(tokenizer.embedded_feature, pad_width, mode='constant', constant_values=0)

# Load the saved model from the directory
transformer.load_state_dict(torch.load(model_directory, map_location=torch.device('cpu')))
with torch.no_grad():
    output = transformer(torch.tensor(input_tens))
    print(output)

video_writer.release()
cap.release()
cv2.destroyAllWindows()
# kalman_filter.showPrediction()