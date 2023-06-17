import cv2
import torch
import numpy as np

from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

Path = 'C:/Users/Computer/Documents/GitHub/CV_BasketAnalysis/'

def detect_objects(frame, model, kalmanFilter):
    detections = model(frame)
    # To show the detection use the line below
    # detections.show()
    image = detections[0].plot()

    for result in detections[0].boxes:
        # Get the object class
        class_index = int(result.cls.item())
        class_name = model.names[class_index]

        # If the object is a Basketball make a prediction of its trajectory
        if class_name == 'basketball':
            # Get the bounding box coordinates
            x,y = result.xywh[0,0:2]
            # Predict the trajectory in the next 12 frames
            for i in range(12):
                filtered_x, filtered_y = track_objects(x.item(), y.item(), kalmanFilter)
                if i==0 or i == 5 or i == 11:
                    image = cv2.circle(detections[0].plot(), (int(filtered_x), int(filtered_y)), 2, (0, 255, 0), thickness=3)
                x,y = filtered_x,filtered_y
    return image

def track_objects(x, y, kalman_filter):
    kalman_filter.x = np.array([x, y, 0, 0]).reshape(4, 1)
    # Generate the new estimate position
    z = np.array([[x + np.random.randn(1)[0] * 2],
                    [y + np.random.randn(1)[0] * 2]])

    # Kalman's Filter prediction
    kalman_filter.predict()

    # Update of the kalman's filter with the new position
    kalman_filter.update(z)

    # Recupero della posizione filtrata della palla
    filtered_state = kalman_filter.x
    filtered_x = filtered_state[0][0]
    filtered_y = filtered_state[1][0]

    return filtered_x, filtered_y

# Loop principale del video
# Load yolov8 model

model = YOLO('yolov8s_custom/weights/best.pt')

# Inizialize Kalman Filter 
kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
kalman_filter.F = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
kalman_filter.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
kalman_filter.P *= 1000
kalman_filter.R = np.array([[0.1, 0],
                            [0, 0.1]])
kalman_filter.Q = np.array([[0.1, 0, 0, 0],
                            [0, 0.1, 0, 0],
                            [0, 0, 0.1, 0],
                            [0, 0, 0, 0.1]])

cap = cv2.VideoCapture(f'{Path}dataset/ours/video/video2.mp4')
ret, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24
frame_width = frame.shape[1]
frame_height = frame.shape[0]
video_writer = cv2.VideoWriter('video_detections.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    
    ret, frame = cap.read()

    if not ret:
        break

    # Rileva e traccia gli oggetti nel frame
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    new_frame = detect_objects(frame, model, kalman_filter)
    video_writer.write(new_frame)

video_writer.release()
cap.release()
cv2.destroyAllWindows()