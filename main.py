import cv2
import torch
import numpy as np

from filterpy.kalman import KalmanFilter

Path = 'C:/Users/Computer/Documents/GitHub/CV_BasketAnalysis/'

def detect_objects(frame, model, kalmanFilter):
    detections = model(frame)
    # To show the detection use the line below
    # detections.show()

    for result in detections.xywh:
        # Get the bounding box coordinates

        for i in range(result.shape[0]):
            # Get the object class
            class_index = int(result[i,5])
            class_name = model.names[class_index]

            # If the object is a Basketball make a prediction of its trajectory
            if class_name == 'basketball':
                x,y = result[i,0:2]
                # Predict the trajectory in the next 12 frames
                for i in range(12):
                    filtered_x, filtered_y = track_objects(x.item(), y.item(), kalmanFilter)
                    if i==0 or i == 5 or i == 11:
                        cv2.circle(detections.render()[0], (int(filtered_x), int(filtered_y)), 2, (0, 255, 0), thickness=3)
                    x,y = filtered_x,filtered_y

    return detections.render()[0]

def track_objects(x, y, kalman_filter):
    kalman_filter.x = np.array([x, y, 0, 0]).reshape(4, 1)
    # detections expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
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
# Load yolov5 model 
model = torch.hub.load(f'{Path}yolo5', 'custom',
                            path=f"{Path}yolo5/runs/train/yolo_basket_det_PDataset/weights/best.pt",source='local', force_reload=True)

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

cap = cv2.VideoCapture(f'{Path}dataset/ours/video/video1.mp4')
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